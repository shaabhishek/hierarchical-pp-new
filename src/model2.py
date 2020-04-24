import math

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Normal, Categorical

from base_model import BaseEncoder, BaseDecoder, BaseModel
from base_model import sample_gumbel_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(BaseEncoder):
    def __init__(self, rnn_dims:list, y_dims:list, z_dims:list, reverse_rnn_dims:list):
        super().__init__(rnn_dims, y_dims, z_dims)
        self.reverse_rnn_dims = reverse_rnn_dims
        self.reverse_cell = nn.GRUCell(input_size=self.reverse_rnn_dims[0], hidden_size=self.reverse_rnn_dims[1])

    def forward(self, xt, h, temp, mask):
        T,BS,_ = xt.shape

        # Compute encoder RNN hidden states for y
        h_0 = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        hidden_seq, _ = self.rnn_module(xt, h_0)
        
        # Encoder for y.  Need the last one based on mask
        last_seq = torch.argmax(mask, dim=0)#Time dimension
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

        # Encoder for z Reverse RNN
        rh_ = torch.zeros(BS, self.rnn_hidden_dim).to(device)
        concat_hx = torch.cat([xt, h], dim = -1) #T x BS x hidden_dim + embedding_dim
        outs = []
        for seq in range(T):
            rh_ = self.reverse_cell(concat_hx[T-1-seq,:,:], rh_)
            #rh_ shape BSxdim. Multiply with mask
            rh_ = rh_ * mask[T-1-seq,:][:,None]
            outs.append(rh_[None,:,:]) #1, BS, dim
        rh = torch.cat(outs, dim = 0)
        #rh = torch.flip(rh , dims = [0])#T, BS, dim

        # Ancestral sampling
        dist_z, sample_z  = [], []
        z = torch.zeros(1, BS, self.latent_dim).to(device)
        for seq in range(T):
            concat_ayz = torch.cat([rh[T-1-seq,:,:][None,:,:], z, sample_y[seq,:,:][None,:,:] ], dim = -1)#1, BS, latent+cluster+hidden_dim
            dist_z_ = self.z_module(concat_ayz)
            sample_z_ = dist_z_.rsample()
            dist_z.append(dist_z_)
            sample_z.append(sample_z_)
            z = sample_z_
        
        mu_z = torch.cat([dist.mean for dist in dist_z], dim =0)
        stddev_z = torch.cat([dist.stddev for dist in dist_z], dim =0)
        dist_z = Normal(loc=mu_z, scale=stddev_z)
        sample_z = torch.cat(sample_z, dim = 0)

        try:
            assert sample_z.shape == (T, BS, self.latent_dim)
        except AssertionError:
            print(sample_z.shape)
            # import pdb; pdb.set_trace()
            raise
        return (sample_y, dist_y), (sample_z, dist_z)
    
class Decoder(BaseDecoder):
    def __init__(self, shared_output_dims:list, marker_dim:int, decoder_in_dim:int, **kwargs):
        super().__init__(shared_output_dims, marker_dim, decoder_in_dim, **kwargs)

    def forward(self, concat_hzy):
        out = self.preprocessing_module(concat_hzy)
        return out

class Model2(BaseModel):
    def __init__(self, **kwargs):
    # def __init__(self, latent_dim=20, marker_dim=31, marker_type='real', rnn_hidden_dim=128, time_dim=2, cluster_dim=5, x_given_t=False, time_loss='normal', gamma=1., dropout=None, base_intensity=0, time_influence=0.1):
        super().__init__(**kwargs)
        self.logvar_min = math.log(1e-20)
        self.sigma_min = 1e-10
        self.gamma = kwargs['gamma']
        self.dropout = kwargs['dropout']
        
        # Forward RNN
        self.rnn = self.create_rnn()

        # Inference network
        # Override the values provided to z_module by base model
        self.reverse_rnn_dims = [self.emb_dim+self.rnn_hidden_dim, self.rnn_hidden_dim] #phi_xt+h -> h
        z_input_dim = self.reverse_rnn_dims[-1] + self.cluster_dim + self.latent_dim #h+y+z
        self.z_dims[0] = z_input_dim
        self.encoder = Encoder(rnn_dims=self.rnn_dims, y_dims=self.y_dims, z_dims=self.z_dims, reverse_rnn_dims=self.reverse_rnn_dims)

        # Generative network
        encoder_out_dim = self.rnn_hidden_dim+self.latent_dim+self.cluster_dim
        decoder_kwargs = {'marker_dim': self.marker_dim, 'decoder_in_dim': encoder_out_dim, 'time_loss': self.time_loss, 'marker_type': self.marker_type, 'x_given_t': self.x_given_t}
        self.decoder = Decoder(shared_output_dims=self.shared_output_dims, **decoder_kwargs)
        create_output_nets(self.decoder, self.base_intensity, self.time_influence)

        #Prior on z
        self.prior_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.rnn_hidden_dim),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(self.rnn_hidden_dim, self.latent_dim)
        self.prior_logvar = nn.Linear(self.rnn_hidden_dim, self.latent_dim)

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
        meta_info = {"marker_nll":marker_loss.detach().cpu(), "time_nll":time_loss.detach().cpu(), "true_ll": true_loss.detach().cpu(), "kl": KL.detach().cpu()}
        return loss, {**meta_info, **metric_dict}

    def prior(self, sample_z):
        #Sample_z is shape T, BS, latent_dim
        T,BS, l_dim = sample_z.shape
        hiddenlayer = self.prior_net(sample_z)
        mu = self.prior_mu(hiddenlayer)
        logvar = torch.clamp(self.prior_logvar(hiddenlayer), min = self.logvar_min)#T,BS, dim
        base_mu = torch.zeros(1,BS, l_dim).to(device)
        base_logvar = torch.zeros(1,BS, l_dim).to(device)
        mu = torch.cat([base_mu, mu[:-1,:,:]], dim =0)
        logvar = torch.cat([base_logvar, logvar[:-1,:,:]], dim =0)
        return mu, logvar
    
    def _forward(self, x, t, temp, mask):
        # Transform markers and timesteps into the embedding spaces
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1)
        T,BS,_ = phi_x.shape

        ##Compute h_t Shape T+1, BS, dim
        # Run RNN over the concatenated embedded sequence
        h_0 = torch.zeros(1, BS, self.rnn_hidden_dim).to(device)
        # Run RNN
        hidden_seq, _ = self.rnn(phi_xt, h_0)
        # Append h_0 to h_1 .. h_T
        hidden_seq = torch.cat([h_0, hidden_seq], dim=0)
                
        ## Inference a_t= q([x_t, h_t], a_{t+1})
        # Get the sampled value and (mean + var) latent variable
        # using the hidden state sequence
        # posterior_sample_y, posterior_sample_z, posterior_logits_y, (posterior_mu_z, posterior_logvar_z) = self.encoder(phi_xt, hidden_seq[:-1, :,:], temp, mask)
        (posterior_sample_y, posterior_dist_y), (posterior_sample_z, posterior_dist_z) = self.encoder(phi_xt, hidden_seq[:-1, :,:], temp, mask)

    
        # Create distributions for Posterior random vars
        # posterior_dist_z = Normal(posterior_mu_z, torch.exp(posterior_logvar_z*0.5))
        # posterior_dist_y = Categorical(logits=posterior_logits_y)
        
        # Prior is just a Normal(0,1) dist for z and Uniform Categorical for y
        #prior dist z is TxBSx latent_dim. T=0=> Normal(0,1)
        prior_mu, prior_logvar = self.prior(posterior_sample_z)##Normal(0, 1)
        prior_dist_z  = Normal(prior_mu, (prior_logvar*0.5).exp())


        
        prior_dist_y = Categorical(probs=1./self.cluster_dim* torch.ones(1,BS, self.cluster_dim).to(device))

        ## Generative Part
        
        # Use the embedded markers and times to create another set of 
        # hidden vectors. Can reuse the h_0 and time_marker combined computed above
        
        # Combine (z_t, h_t, y) form the input for the generative part
        concat_hzy = torch.cat([hidden_seq[:-1], posterior_sample_z, posterior_sample_y], dim=-1)
        # phi_hzy = self.gen_pre_module(concat_hzy)
        # mu_marker, logvar_marker = generate_marker(self, phi_hzy, None)
        # time_log_likelihood, mu_time = compute_point_log_likelihood(self, phi_hzy, t)
        # marker_log_likelihood = compute_marker_log_likelihood(self, x, mu_marker, logvar_marker)
        
        phi_hzy = self.decoder(concat_hzy)

        dist_marker_recon = self.decoder.generate_marker(phi_hzy, t)
        time_log_likelihood, mu_time = self.decoder.compute_time_log_prob(phi_hzy, t) #(T,BS)
        marker_log_likelihood = self.decoder.compute_marker_log_prob(x, dist_marker_recon) #(T,BS)

        KL_cluster = kl_divergence(posterior_dist_y, prior_dist_y)
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