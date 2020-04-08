import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal, Categorical

from base_model import compute_marker_log_likelihood, compute_point_log_likelihood, generate_marker, create_output_nets, sample_gumbel_softmax
from base_model import MLP, MLPNormal, MLPCategorical, BaseEncoder, BaseDecoder, BaseModel
from utils.metric import get_marker_metric, compute_time_expectation, get_time_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mlp(dims:list):
    layers = list()
    for i in range(len(dims) - 1):
        n = dims[i]
        m = dims[i+1]
        L = nn.Linear(n, m, bias=True)
        layers.append(L)
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)

class Encoder(BaseEncoder):
    def __init__(self, rnn_dims:list, y_dims:list, z_dims:list, ):
        super().__init__(rnn_dims, y_dims, z_dims)

    def forward(self, xt, temp, mask):
        """
        Input:
            xt: Tensor of shape T x BS x (self.x_embedding_dim[-1]+self.t_embedding_dim[-1])
            temp: scalar
            mask : Tensor TxBS
        Output:
            sample_y: Tensor of shape T x BS x cluster_dim
            sample_z: Tensor of shape T x BS x latent_dim
            logits_y: Tensor of shape 1 x BS x cluster_dim
            mu_z: Tensor of shape T x BS x latent_dim
            logvar_z: Tensor of shape T x BS x latent_dim
        """
        T,BS,_ = xt.shape

        # Compute encoder RNN hidden states
        h_0 = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        hidden_seq, _ = self.rnn_module(xt, h_0) #(T, BS, rnn_hidden_dim)

        # Encoder for y.  Need the last one based on mask
        last_seq = torch.argmax(mask, dim=0)#Time dimension - (BS,)
        # Pick out the state corresponding to last time step for each batch data point
        final_state = torch.cat([hidden_seq[last_seq[i],i][None, :] for i in range(BS)], dim = 0) #(BS,rnn_hidden_dim)
        try:
            assert final_state.shape == (BS, self.rnn_hidden_dim)
        except AssertionError:
            print(final_state.shape)
            raise
        
        dist_y = self.y_module(final_state) #shape(dist_y.logits) = (BS, y_dim)
        sample_y = sample_gumbel_softmax(dist_y.logits, temp) #(BS, y_dim)
        sample_y = sample_y.unsqueeze(0).expand(T,-1,-1) #(T,BS,y_dim)
        
        try:
            assert sample_y.shape == (T, BS, self.y_dim)
        except AssertionError:
            print(sample_y.shape)
            # import pdb; pdb.set_trace()
            raise

        # Encoder for z
        hidden_seq = torch.cat([h_0, hidden_seq], dim=0)
        concat_hxty = torch.cat([hidden_seq[:-1], xt, sample_y], dim=-1) #(T, BS, ..)

        dist_z = self.z_module(concat_hxty) #(T,BS,latent_dim)
        sample_z = dist_z.rsample() #(T,BS,latent_dim)
        return (sample_y, dist_y), (sample_z, dist_z)

class Decoder(BaseDecoder):
    def __init__(self, shared_output_dims:list, marker_dim:int, decoder_in_dim:int, **kwargs):
        super().__init__(shared_output_dims, marker_dim, decoder_in_dim, **kwargs)

    def forward(self, concat_hzy):
        out = self.preprocessing_module(concat_hzy)
        return out

class Model1(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigma_min = 1e-10
        self.gamma = kwargs['gamma']
        self.dropout = kwargs['dropout']

        # Forward RNN
        self.rnn = self.create_rnn()

        # Inference network
        self.encoder = Encoder(rnn_dims=self.rnn_dims, y_dims=self.y_dims, z_dims=self.z_dims)

        # Generative network
        encoder_out_dim = self.rnn_hidden_dim+self.latent_dim+self.cluster_dim
        decoder_kwargs = {'marker_dim': self.marker_dim, 'decoder_in_dim': encoder_out_dim, 'time_loss': self.time_loss, 'marker_type': self.marker_type, 'x_given_t': self.x_given_t}
        self.decoder = Decoder(shared_output_dims=self.shared_output_dims, **decoder_kwargs)
        create_output_nets(self.decoder, self.base_intensity, self.time_influence)

    def create_rnn(self):
        rnn = nn.GRU(
            input_size=self.x_embedding_dim[-1]+self.t_embedding_dim[-1],
            hidden_size=self.rnn_hidden_dim,
        )
        return rnn

    def create_output_nets(self):
        l = self.shared_output_dims[-1]
        t_module_mu = nn.Linear(l, 1)
        t_module_logvar = nn.Linear(l, 1)

        x_module_logvar = None
        if self.x_given_t:
            l += 1
        if self.marker_type == 'real':
            x_module_mu = nn.Linear(l, self.marker_dim)
            x_module_logvar = nn.Linear(l, self.marker_dim)
        elif self.marker_type == 'binary':#Fix binary
            x_module_mu = nn.Sequential(
                nn.Linear(l, self.marker_dim),
                nn.Sigmoid())
        elif self.marker_type == 'categorical':
            x_module_mu = nn.Sequential(
                nn.Linear(l, self.marker_dim)#,
                #nn.Softmax(dim=-1)
            )
        return t_module_mu, t_module_logvar, x_module_mu, x_module_logvar

    def forward(self, marker_seq, time_seq, anneal=1., mask=None, temp=0.5):

        time_log_likelihood, marker_log_likelihood, KL, metric_dict = self._forward(marker_seq, time_seq, temp, mask)

        marker_loss = (-1.* marker_log_likelihood *mask)[1:,:].sum()
        time_loss = (-1. *time_log_likelihood *mask)[1:,:].sum()

        NLL = self.gamma*time_loss + marker_loss
        loss = NLL + anneal*KL
        true_loss = time_loss + marker_loss +KL
        meta_info = {"marker_ll":marker_loss.detach().cpu(), "time_ll":time_loss.detach().cpu(), "true_ll": true_loss.detach().cpu(), "kl": KL.detach().cpu()}
        return loss, {**meta_info, **metric_dict}

    def _forward(self, x, t, temp, mask):
        # Transform markers and timesteps into the embedding spaces
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1) #(T,BS, emb_dim)
        T,BS,_ = phi_x.shape

        ## Inference
        # Get the sampled value and (mean + var) latent variable
        # using the hidden state sequence
        (posterior_sample_y, posterior_dist_y), (posterior_sample_z, posterior_dist_z) = self.encoder(phi_xt, temp, mask)

        # Create distributions for Posterior random vars
        # posterior_dist_z = Normal(posterior_mu_z, torch.exp(posterior_logvar_z*0.5))
        # posterior_dist_y = Categorical(logits=posterior_logits_y)

        # Prior is just a Normal(0,1) dist for z and Uniform Categorical for y
        prior_dist_z = Normal(torch.tensor([0.]).to(device), torch.tensor([1.]).to(device))

        prior_dist_y = Categorical(logits=torch.ones(1,BS, self.cluster_dim).to(device))

        ## Generative Part

        # Use the embedded markers and times to create another set of
        # hidden vectors. Can reuse the h_0 and time_marker combined computed above

        # Run RNN over the concatenated embedded sequence
        h_0 = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        # Run RNN
        hidden_seq, _ = self.rnn(phi_xt, h_0)
        # Append h_0 to h_1 .. h_T
        hidden_seq = torch.cat([h_0, hidden_seq], dim=0)

        # Combine (z_t, h_t, y) form the input for the generative part
        concat_hzy = torch.cat([hidden_seq[:-1], posterior_sample_z, posterior_sample_y], dim=-1) #(T,BS,h+z+y dims)
        phi_hzy = self.decoder(concat_hzy) #(T,BS,shared_output_dims[-1])
        
        dist_marker_recon = self.decoder.generate_marker(phi_hzy, t)
        time_log_likelihood, mu_time = self.decoder.compute_time_log_prob(phi_hzy, t) #(T,BS)
        marker_log_likelihood = self.decoder.compute_marker_log_prob(x, dist_marker_recon) #(T,BS)

        KL_cluster = kl_divergence(posterior_dist_y, prior_dist_y) #(1,BS)
        KL_z = kl_divergence(posterior_dist_z, prior_dist_z).sum(-1)*mask
        KL = KL_cluster.sum() + KL_z.sum()
        try:
            assert (KL >= 0)
        except:
            raise ValueError("KL should be non-negative")

        metric_dict = {"z_cluster":0}
        with torch.no_grad():
            # Metric not needed
            """
            if self.time_loss == 'intensity':
                mu_time = compute_time_expectation(self, phi_hzy, t, mask)[:,:, None]
            get_marker_metric(self.marker_type, mu_marker, x, mask, metric_dict)
            get_time_metric(mu_time,  t, mask, metric_dict)
            """
            metric_dict['marker_acc'] = -1.
            metric_dict['marker_acc_count']  = 1.
            metric_dict['time_mse'] = 1.
            metric_dict['time_mse_count'] = 1.

        return time_log_likelihood, marker_log_likelihood, KL, metric_dict