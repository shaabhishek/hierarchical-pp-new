import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal, Categorical
import math
import numpy as np

from base_model import compute_marker_log_likelihood, compute_point_log_likelihood, generate_marker, create_output_nets
from utils.metric import get_marker_metric, compute_time_expectation, get_time_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel(shape, eps=1e-20):
    unif = torch.rand(*shape).to(device)
    g = -torch.log(-torch.log(unif + eps) + eps)
    return g


def sample_gumbel_softmax(logits, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar

        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = sample_gumbel(logits.shape)
    assert g.shape == logits.shape
    h = (g + logits)/temperature
    y = F.softmax(h, dim=-1)
    return y


def reparameterize(mu, logvar):
    epsilon = torch.randn_like(mu).to(device)
    sigma = torch.exp(0.5 * logvar)
    return mu + epsilon.mul(sigma)


class Model2Filter(nn.Module):
    def __init__(self,n_sample , latent_dim=20, marker_dim=31, marker_type='real', hidden_dim=128, time_dim=2, n_cluster=5, x_given_t=False, time_loss='normal', gamma=1., dropout=None, base_intensity=0., time_influence=0.1):
        super().__init__()
        self.marker_type = marker_type
        self.marker_dim = marker_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cluster_dim = n_cluster
        self.x_given_t = x_given_t
        self.time_loss = time_loss
        self.logvar_min = math.log(1e-10)
        self.sigma_min = 1e-10
        self.gamma = gamma
        self.dropout = dropout
        self.n_sample = n_sample

        # Preprocessing networks
        # Embedding network
        self.x_embedding_layer = [128]
        self.t_embedding_layer = [8]
        self.embed_x, self.embed_t=self.create_embedding_nets()
        self.shared_output_layers=[256]
        self.inf_pre_module, self.gen_pre_module=self.create_preprocess_nets()

        # Forward RNN
        self.rnn=self.create_rnn()

        # Inference network
        self.encoder_layers=[64, 64]
        self.y_encoder, self.encoder_rnn, self.z_intmd_module, self.z_mu_module, self.z_logvar_module=self.create_inference_nets()

        # Generative network
        create_output_nets(self, base_intensity, time_influence)

        # Prior on z
        self.prior_net=nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.prior_mu=nn.Linear(self.hidden_dim, self.latent_dim)
        self.prior_logvar=nn.Linear(self.hidden_dim, self.latent_dim)

    def create_embedding_nets(self):
        # marker_dim is passed. timeseries_dim is 2
        if self.marker_type == 'categorical':
            x_module = nn.Embedding(self.marker_dim, self.x_embedding_layer[0])
        else:
            x_module = nn.Sequential(
                nn.Linear(self.marker_dim, self.x_embedding_layer[0]),
                nn.ReLU(),
            )

        t_module = nn.Sequential(
            nn.Linear(self.time_dim, self.t_embedding_layer[0]),
            nn.ReLU()
        )
        return x_module, t_module

    def create_preprocess_nets(self):
        # Inference net preprocessing
        hxty_input_dim = self.hidden_dim+self.latent_dim+self.cluster_dim + \
            self.x_embedding_layer[-1] + self.t_embedding_layer[-1]
        inf_pre_module = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(hxty_input_dim, hxty_input_dim),
            nn.ReLU(), nn.Dropout(self.dropout))

        # Generative net preprocessing
        hzy_input_dim = self.hidden_dim+self.latent_dim+self.cluster_dim
        gen_pre_module = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(hzy_input_dim, self.shared_output_layers[-1]),
            nn.ReLU(), nn.Dropout(self.dropout))
        return inf_pre_module, gen_pre_module

    def create_rnn(self):
        rnn = nn.GRU(
            input_size = self.x_embedding_layer[-1]+self.t_embedding_layer[-1],
            hidden_size = self.hidden_dim,
        )
        return rnn

    def create_inference_nets(self):
        y_module = nn.Sequential(
            nn.Linear(self.hidden_dim, self.cluster_dim)
        )

        encoder_rnn = nn.GRU(
            input_size = self.x_embedding_layer[-1]+self.t_embedding_layer[-1],
            hidden_size = self.hidden_dim,
        )

        z_input_dim = self.hidden_dim+self.latent_dim+self.cluster_dim + \
            self.x_embedding_layer[-1] + self.t_embedding_layer[-1]
        z_intmd_module = nn.Sequential(
            nn.Linear(z_input_dim, self.encoder_layers[0]),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[0], self.encoder_layers[1]),
            nn.ReLU(),
        )
        z_mu_module = nn.Linear(self.encoder_layers[1], self.latent_dim)
        z_logvar_module = nn.Linear(self.encoder_layers[1], self.latent_dim)
        return y_module, encoder_rnn, z_intmd_module, z_mu_module, z_logvar_module




    ### ENCODER ###
    def encoder(self, phi_xt, h_t, temp, mask, n_sample):
        """
        Input:
            phi_xt: Tensor of shape T x BS x (self.x_embedding_layer[-1]+self.t_embedding_layer[-1])
            h_t : T x BS x hidden_dim
            temp: scalar
            mask : Tensor TxBS
        Output:
            sample_filter_y: Tensor of shape T x BS x cluster_dim for filtering
            sample_filter_z: Tensor of shape T x BS x latent_dim
            logits_y: Tensor of shape T x BS x cluster_dim for filtering
            mu_z: Tensor of shape T x BS x latent_dim
            logvar_z: Tensor of shape T x BS x latent_dim
            pred_y : T x n_sample x BS x cluster_dim for prediction
            pred_z : T x n_sample x BS x latent_dim for prediction

        """
        T, BS, _=phi_xt.shape

        # Compute encoder RNN hidden states for y
        h_0=torch.zeros(1, BS, self.hidden_dim).to(device)
        hidden_seq, _=self.encoder_rnn(phi_xt, h_0)
        # T+1 x BS x hidden_dim
        hidden_seq=torch.cat([h_0, hidden_seq], dim=0)
        logits_y=self.y_encoder(hidden_seq)[
            :, None, :, :]  # T+1 x 1 x BS x k
        # n_sample and pred
        repeat_vals=(-1, n_sample, -1, -1)
        logits_y=logits_y.expand(*repeat_vals)  # T+1 x n_sample x BS x k
        sample_y=sample_gumbel_softmax(
            logits_y, temp)  # T+1 x n_sample x BS x k
        sample_pred_y=sample_y[:-1, :, :, :]  # T x sample x BS x k

        logits_filter_y=logits_y[1:, :, :, :]  # T x n_sample x BS x K
        # T x BS x K used for filtering loglikelihood
        sample_filter_y=sample_y[1:, :, :, :].contiguous().view(
            T, BS * n_sample, -1)  # T x n_sample * BS x k

        # T x BS x hidden_dim + embedding_dim
        concat_hx=torch.cat([phi_xt, h_t], dim=-1)
        mu_z, logvar_z, sample_z=[], [], []
        z=torch.zeros(1, BS, self.latent_dim).to(device)
        for seq in range(T):
            concat_ayz = torch.cat(
                [concat_hx[seq, :, :][None, :, :], z, sample_y[seq, 0,:, :][None, :, :]], dim = -1)
            phi_ayz = self.inf_pre_module(concat_ayz)  # 1, BS, ...
            z_intmd = self.z_intmd_module(phi_ayz)
            mu_z_ = self.z_mu_module(z_intmd)
            logvar_z_ = self.z_logvar_module(z_intmd)
            sample_z_ = reparameterize(mu_z_, logvar_z_)
            mu_z.append(mu_z_)
            logvar_z.append(logvar_z_)
            sample_z.append(sample_z_)
            z = sample_z_

        ret_mu_z = torch.cat(mu_z, dim = 0)  # .view(T, n_sample, BS, -1)
        ret_logvar_z = torch.cat(logvar_z, dim = 0)  # .view(T, n_sample, BS, -1)
        ret_sample_z = torch.cat(sample_z, dim = 0)  # .view(T, n_sample, BS, -1)
        


        
        #For prediction
        if self.training == False:
            concat_hx_pred=concat_hx[:, None, :, :].expand(
                *repeat_vals)  # (T, BS ,n_sample, -1)
            concat_hx_pred=concat_hx_pred.contiguous().view(
                T, BS * n_sample, -1)  # T x n_sample * BS x dim
            mu_z, logvar_z, sample_z=[], [], []
            z=torch.zeros(1, BS*n_sample, self.latent_dim).to(device)


            for seq in range(T):
                # 1, BS, latent+cluster+hidden_dim+embedding_dim
                concat_ayz = torch.cat(
                    [concat_hx_pred[seq, :, :][None, :, :], z, sample_filter_y[seq, :, :][None, :, :]], dim = -1)
                phi_ayz = self.inf_pre_module(concat_ayz)  # 1, BS, ...
                z_intmd = self.z_intmd_module(phi_ayz)
                mu_z_ = self.z_mu_module(z_intmd)
                logvar_z_ = self.z_logvar_module(z_intmd)
                sample_z_ = reparameterize(mu_z_, logvar_z_)
                mu_z.append(mu_z_)
                logvar_z.append(logvar_z_)
                sample_z.append(sample_z_)
                z = sample_z_

            mu_z = torch.cat(mu_z, dim = 0)  # .view(T, n_sample, BS, -1)
            logvar_z = torch.cat(logvar_z, dim = 0)  # .view(T, n_sample, BS, -1)
            sample_z = torch.cat(sample_z, dim = 0)  # .view(T, n_sample, BS, -1)

            # Prior Distribution For Prediction
            # Prior mu, logvar T x n_sample x BS x latent_dim
            prior_mu, prior_logvar=self.prior(sample_z)
            # T x sample x BS x latent_dim
            pred_z=reparameterize(prior_mu, prior_logvar)
            pred_z=pred_z.view(T, n_sample, BS, -1)

            # # Reshape
            # sample_filter_y=sample_filter_y.view(T, n_sample, BS, -1)
            # sample_z=sample_z.view(T, n_sample, BS, -1)
            # mu_z=mu_z.view(T, n_sample, BS, -1)
            # logvar_z=logvar_z.view(T, n_sample, BS, -1)
        else:
            sample_pred_y, pred_z = None, None

        return sample_y[1:, 0, :, :], ret_sample_z[:, :, :], logits_filter_y[:, 0, :, :], (ret_mu_z[:, :, :], ret_logvar_z[:,  :, :]), sample_pred_y, pred_z

    def forward(self, marker_seq, time_seq, anneal=1., mask=None, temp=0.5):
        time_log_likelihood, marker_log_likelihood, KL, metric_dict=self._forward(
            marker_seq, time_seq, temp, mask)

        marker_loss=(-1. * marker_log_likelihood * mask)[1:, :].sum()
        time_loss=(-1. * time_log_likelihood * mask)[1:, :].sum()

        NLL=self.gamma*time_loss + marker_loss
        loss=NLL + anneal*KL
        true_loss=time_loss + marker_loss + KL
        meta_info={"marker_ll": marker_loss.detach().cpu(), "time_ll": time_loss.detach(
        ).cpu(), "true_ll": true_loss.detach().cpu(), "kl": KL.detach().cpu()}
        return loss, {**meta_info, **metric_dict}

    def prior(self, sample_z):
        # Sample_z is shape T, BS, latent_dim
        T, BS, l_dim=sample_z.shape
        hiddenlayer=self.prior_net(sample_z)
        mu=self.prior_mu(hiddenlayer)
        logvar=torch.clamp(self.prior_logvar(hiddenlayer),
                             min=self.logvar_min)  # T,BS, dim
        base_mu=torch.zeros(1, BS, l_dim).to(device)
        base_logvar=torch.zeros(1, BS, l_dim).to(device)
        mu=torch.cat([base_mu, mu[:-1, :, :]], dim=0)
        logvar=torch.cat([base_logvar, logvar[:-1, :, :]], dim=0)
        return mu, logvar

    def _forward(self, x, t, temp, mask):
        n_sample= self.n_sample
        batch_len = mask.sum(dim= 0, keepdim = True)
        # Transform markers and timesteps into the embedding spaces
        phi_x, phi_t=self.embed_x(x), self.embed_t(t)
        phi_xt=torch.cat([phi_x, phi_t], dim=-1)
        T, BS, _=phi_x.shape

        # Compute h_t Shape T+1, BS, dim
        # Run RNN over the concatenated embedded sequence
        h_0=torch.zeros(1, BS, self.hidden_dim).to(device)
        # Run RNN
        hidden_seq, _=self.rnn(phi_xt, h_0)
        # Append h_0 to h_1 .. h_T
        hidden_seq=torch.cat([h_0, hidden_seq], dim=0)

        # Inference a_t= q([x_t, h_t], a_{t+1})
        # Get the sampled value and (mean + var) latent variable
        # using the hidden state sequence
        posterior_sample_y, posterior_sample_z, posterior_logits_y, (
            posterior_mu_z, posterior_logvar_z), pred_y, pred_z=self.encoder(phi_xt, hidden_seq[:-1, :, :], temp, mask, n_sample=n_sample)

        # Create distributions for Posterior random vars
        posterior_dist_z=Normal(
            posterior_mu_z, torch.exp(posterior_logvar_z*0.5))
        posterior_dist_y=Categorical(logits=posterior_logits_y)

        # Prior is just a Normal(0,1) dist for z and Uniform Categorical for y
        # prior dist z is TxBSx latent_dim. T=0=> Normal(0,1)
        prior_mu, prior_logvar=self.prior(posterior_sample_z)  # Normal(0, 1)
        prior_dist_z=Normal(prior_mu, (prior_logvar*0.5).exp())

        prior_dist_y=Categorical(
            probs=1./self.cluster_dim * torch.ones(T, BS, self.cluster_dim).to(device))

        # Generative Part

        # Use the embedded markers and times to create another set of
        # hidden vectors. Can reuse the h_0 and time_marker combined computed above

        # Combine (z_t, h_t, y) form the input for the generative part
        concat_hzy=torch.cat(
            [hidden_seq[:-1], posterior_sample_z, posterior_sample_y], dim=-1)
        phi_hzy=self.gen_pre_module(concat_hzy)
        mu_marker, logvar_marker=generate_marker(self, phi_hzy, None)
        time_log_likelihood, _ =compute_point_log_likelihood(
            self, phi_hzy, t)
        marker_log_likelihood=compute_marker_log_likelihood(
            self, x, mu_marker, logvar_marker)

        KL_cluster=kl_divergence(posterior_dist_y, prior_dist_y) * mask
        KL_z=kl_divergence(posterior_dist_z, prior_dist_z).sum(-1)*mask
        KL=(KL_cluster/batch_len).sum() + KL_z.sum()
        # try:
        #     assert (KL >= 0)
        # except:
        #     import pdb
        #     pdb.set_trace()

        metric_dict={"z_cluster": 0}
        if self.training:
            metric_dict['marker_acc'] = -1.
            metric_dict['marker_acc_count']  = 1.
            metric_dict['time_mse'] = 1.
            metric_dict['time_mse_count'] = 1.
        else:
            with torch.no_grad():
                # Prediction Layer # T x 10 x BS x dim
                pred_h = hidden_seq[:-1][:, None, :, :]  # T x 1 x BS x dim
                repeat_vals = (-1, n_sample, -1, -1)
                pred_h = pred_h.expand(*repeat_vals)  # T x n_sample x BS x dim
                pred_hzy = torch.cat([pred_h, pred_z, pred_y], dim = -1)
                pred_hzy = self.gen_pre_module(pred_hzy)
                pred_mu_marker, _=generate_marker(self, pred_hzy, None)  # T x n_sample x BS x dim
                if self.time_loss == 'intensity':
                    pred_mu_time = compute_time_expectation(
                        self, pred_hzy, t, mask)[:, :, None]
                else:
                    pred_mu_time=self.time_mu(pred_hzy)  # TxsamplexBSx1

                get_marker_metric(self.marker_type, pred_mu_marker,
                                x, mask, metric_dict)
                get_time_metric(pred_mu_time,  t, mask, metric_dict)
                # if preds_file is not None:
                #     import pdb; pdb.set_trace()
                #     if len(pred_mu_time.data.size()) == 3:
                #         np.savetxt(preds_file, (pred_mu_time[1:,:,0]*mask[1:, :]).cpu().numpy().T)
                #     else:
                #         np.savetxt(preds_file, (torch.mean(pred_mu_time, dim =1)[1:,:,0]*mask[1:, :]).cpu().numpy().T)



        return time_log_likelihood, marker_log_likelihood, KL, metric_dict
