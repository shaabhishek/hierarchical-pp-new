import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal, Categorical

from base_model import compute_marker_log_likelihood, compute_point_log_likelihood, generate_marker
from utils.metric import get_marker_metric, compute_time_expectation, get_time_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_gumbel(shape, eps=1e-20):
    unif = torch.rand(*shape).to(device)
    g = -torch.log(-torch.log(unif + eps))
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
    
    
class Model1(nn.Module):
    def __init__(self, latent_dim=20, marker_dim=31, marker_type='real', hidden_dim=128, time_dim=2, n_cluster=5, x_given_t=False, time_loss='normal', gamma=1., dropout=None, base_intensity=None, time_influence=None):
        super().__init__()
        self.marker_type = marker_type
        self.marker_dim = marker_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cluster_dim = n_cluster
        self.x_given_t = x_given_t
        self.time_loss = time_loss
        self.sigma_min = 1e-2
        self.gamma = gamma
        self.dropout = dropout
        
        # Preprocessing networks
        # Embedding network
        self.x_embedding_layer = [128]
        self.t_embedding_layer = [8]
        self.embed_x, self.embed_t = self.create_embedding_nets()
        self.shared_output_layers = [256]
        self.inf_pre_module, self.gen_pre_module = self.create_preprocess_nets()
        
        # Forward RNN
        self.rnn = self.create_rnn()
        
        # Inference network
        self.encoder_layers = [64, 64]
        self.y_encoder, self.encoder_rnn, self.z_intmd_module, self.z_mu_module, self.z_var_module = self.create_inference_nets()
        
        # Generative network
        self.time_mu, self.time_logvar, self.output_x_mu, self.output_x_logvar = self.create_output_nets()
    
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
        hxty_input_dim = self.hidden_dim+self.x_embedding_layer[-1]+self.t_embedding_layer[-1]+self.cluster_dim
        inf_pre_module = nn.Sequential(
            nn.ReLU(),nn.Dropout(self.dropout),
            nn.Linear(hxty_input_dim, hxty_input_dim),
            nn.ReLU(),nn.Dropout(self.dropout))
        
        # Generative net preprocessing
        hzy_input_dim = self.hidden_dim+self.latent_dim+self.cluster_dim
        gen_pre_module = nn.Sequential(
            nn.ReLU(),nn.Dropout(self.dropout),
            nn.Linear(hzy_input_dim, self.shared_output_layers[-1]),
            nn.ReLU(),nn.Dropout(self.dropout))
        return inf_pre_module, gen_pre_module
        
    
    def create_rnn(self):
        rnn = nn.GRU(
            input_size=self.x_embedding_layer[-1]+self.t_embedding_layer[-1],
            hidden_size=self.hidden_dim,
        )
        return rnn
    
    def create_inference_nets(self):
        y_module = nn.Sequential(
            nn.Linear(self.hidden_dim, self.cluster_dim),
            nn.LogSoftmax(dim=-1)
        )
        
        encoder_rnn = nn.GRU(
            input_size=self.x_embedding_layer[-1]+self.t_embedding_layer[-1],
            hidden_size=self.hidden_dim,
        )
        
        z_input_dim = self.hidden_dim+self.x_embedding_layer[-1]+self.t_embedding_layer[-1]+self.cluster_dim
        z_intmd_module = nn.Sequential(
            nn.Linear(z_input_dim, self.encoder_layers[0]),
            nn.ReLU(),
            nn.Linear(self.encoder_layers[0], self.encoder_layers[1]),
            nn.ReLU(),
        )
        z_mu_module = nn.Linear(self.encoder_layers[1], self.latent_dim)
        z_var_module = nn.Sequential(
            nn.Linear(self.encoder_layers[1], self.latent_dim),
            nn.Softplus(),
        )
        return y_module, encoder_rnn, z_intmd_module, z_mu_module, z_var_module

    def create_output_nets(self):
        l = self.shared_output_layers[-1]
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
    
    ### ENCODER ###
    def encoder(self, phi_xt, temp):
        """
        Input:
            phi_xt: Tensor of shape T x BS x (self.x_embedding_layer[-1]+self.t_embedding_layer[-1])
            temp: scalar
        Output:
            sample_y: Tensor of shape T x BS x cluster_dim
            sample_z: Tensor of shape T x BS x latent_dim
            logits_y: Tensor of shape 1 x BS x cluster_dim
            mu_z: Tensor of shape T x BS x latent_dim
            logvar_z: Tensor of shape T x BS x latent_dim
        """
        T,BS,_ = phi_xt.shape

        # Compute encoder RNN hidden states
        h_0 = torch.zeros(1, BS, self.hidden_dim).to(device)
        hidden_seq, _ = self.encoder_rnn(phi_xt, h_0)
        hidden_seq = torch.cat([h_0, hidden_seq], dim=0)
        
        # Encoder for y
        logits_y = self.y_encoder(hidden_seq[-1])[None, :, :] #shape(logits_y) = 1 x BS x k
        #shape(sample_y) = 1 x BS x k. Should tend to one-hot in the last dimension
        sample_y = sample_gumbel_softmax(logits_y, temp)
        repeat_vals = (T, -1,-1)
        sample_y = sample_y.expand(*repeat_vals) #T x BS x k
        
        # Encoder for z
        concat_hxty = torch.cat([hidden_seq[:-1], phi_xt, sample_y], dim=-1)
        phi_hxty = self.inf_pre_module(concat_hxty)
        z_intmd = self.z_intmd_module(phi_hxty)
        mu_z = self.z_mu_module(z_intmd)
        var_z = self.z_var_module(z_intmd)
        logvar_z = torch.log(var_z + self.sigma_min)
        sample_z = reparameterize(mu_z, logvar_z)
        return sample_y, sample_z, logits_y, (mu_z, logvar_z)
    
    def forward(self, marker_seq, time_seq, anneal=1., mask=None, temp=0.5):
        time_log_likelihood, marker_log_likelihood, KL, metric_dict = self._forward(marker_seq, time_seq, temp, mask)

        marker_loss = (-1.* marker_log_likelihood *mask)[1:,:].sum()
        time_loss = (-1. *time_log_likelihood *mask)[1:,:].sum()
        
        NLL = self.gamma*time_loss + marker_loss
        loss = NLL + KL
        true_loss = time_loss + marker_loss
        meta_info = {"marker_nll":marker_loss.detach().cpu(), "time_nll":time_loss.detach().cpu(), "true_ll": true_loss.detach().cpu(), "kl": KL.detach().cpu()}
        return loss, {**meta_info, **metric_dict}
    
    def _forward(self, x, t, temp, mask):
        # Transform markers and timesteps into the embedding spaces
        phi_x, phi_t = self.embed_x(x), self.embed_t(t)
        phi_xt = torch.cat([phi_x, phi_t], dim=-1)
        T,BS,_ = phi_x.shape
                
        ## Inference
        # Get the sampled value and (mean + var) latent variable
        # using the hidden state sequence
        posterior_sample_y, posterior_sample_z, posterior_logits_y, (posterior_mu_z, posterior_logvar_z) = self.encoder(phi_xt, temp)

        repeat_vals = (T, -1,-1)
        posterior_logits_y = posterior_logits_y.expand(*repeat_vals)
        # Create distributions for Posterior random vars
        posterior_dist_z = Normal(posterior_mu_z, torch.exp(posterior_logvar_z*0.5))
        posterior_dist_y = Categorical(logits=posterior_logits_y)
        
        # Prior is just a Normal(0,1) dist for z and Uniform Categorical for y
        prior_dist_z = Normal(0.*posterior_mu_z, 1. + 0.*posterior_mu_z)
        prior_dist_y = Categorical(probs=1/self.cluster_dim + 0.*posterior_logits_y)

        ## Generative Part
        
        # Use the embedded markers and times to create another set of 
        # hidden vectors. Can reuse the h_0 and time_marker combined computed above

        # Run RNN over the concatenated embedded sequence
        h_0 = torch.zeros(1, BS, self.hidden_dim).to(device)
        # Run RNN
        hidden_seq, _ = self.rnn(phi_xt, h_0)
        # Append h_0 to h_1 .. h_T
        hidden_seq = torch.cat([h_0, hidden_seq], dim=0)
        
        # Combine (z_t, h_t, y) form the input for the generative part
        concat_hzy = torch.cat([hidden_seq[:-1], posterior_sample_z, posterior_sample_y], dim=-1)
        phi_hzy = self.gen_pre_module(concat_hzy)
        mu_marker, logvar_marker = generate_marker(self, phi_hzy, None)
        time_log_likelihood, mu_time = compute_point_log_likelihood(self, phi_hzy, t)
        marker_log_likelihood = compute_marker_log_likelihood(self, x, mu_marker, logvar_marker)
        
        KL_cluster = kl_divergence(posterior_dist_y, prior_dist_y)*mask
        KL_z = kl_divergence(posterior_dist_z, prior_dist_z).sum(-1)*mask
        KL = KL_cluster.sum() + KL_z.sum()
        try:
            assert (KL >= 0)
        except:
            import pdb; pdb.set_trace()
        metric_dict = {}
        with torch.no_grad():
            if self.time_loss == 'intensity':
                mu_time = compute_time_expectation(self, hidden_seq, t, mask)[:,:, None]
            get_marker_metric(self.marker_type, mu_marker, x, mask, metric_dict)
            get_time_metric(mu_time,  t, mask, metric_dict)
            
        return time_log_likelihood, marker_log_likelihood, KL, metric_dict