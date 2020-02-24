import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal, Categorical, Gumbel

from base_model import compute_marker_log_likelihood, compute_point_log_likelihood, generate_marker,create_output_nets
from utils.metric import get_marker_metric, compute_time_expectation, get_time_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_gumbel_softmax(logits, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar

        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = Gumbel(torch.zeros(*logits.shape),torch.ones(*logits.shape)).sample()
    # g = sample_gumbel(logits.shape)
    # assert g.shape == logits.shape
    h = (g + logits)/temperature
    y = F.softmax(h, dim=-1)
    return y

def reparameterize(mu, logvar):
        epsilon = torch.randn_like(mu).to(device)
        sigma = torch.exp(0.5 * logvar)
        return mu + epsilon.mul(sigma)

def create_mlp(dims:list):
    layers = list()
    for i in range(len(dims) - 1):
        n = dims[i]
        m = dims[i+1]
        L = nn.Linear(n, m, bias=True)
        layers.append(L)
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, dims:list):
        assert len(dims) >= 2 #should at least be [inputdim, outputdim]
        super().__init__()
        layers = list()
        for i in range(len(dims) - 1):
            n = dims[i]
            m = dims[i+1]
            L = nn.Linear(n, m, bias=True)
            layers.append(L)
            layers.append(nn.ReLU()) #NOTE: Always slaps a non-linearity in the end
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPNormal(MLP):
    def __init__(self, dims:list):
        try:
            assert len(dims) >= 3 #should at least be [inputdim, hiddendim1, outdim]
        except AssertionError:
            print(dims)
            raise
        
        super().__init__(dims[:-1]) #initializes the core network
        self.mu_module = nn.Linear(dims[-2], dims[-1], bias=False)
        self.logvar_module = nn.Linear(dims[-2], dims[-1], bias=False)

    def forward(self, x):
        h = self.net(x)
        mu, logvar = self.mu_module(h), self.logvar_module(h)
        dist = Normal(mu, logvar.div(2).exp()) #std = exp(logvar/2)
        return dist

class MLPCategorical(MLP):
    def __init__(self, dims:list):
        try:
            assert len(dims) >= 3 #should at least be [inputdim, hiddendim1, logitsdim] - otherwise it's just a matrix multiplication
        except AssertionError:
            print(dims)
            raise
        
        super().__init__(dims[:-1]) #initializes the core network
        self.logit_module = nn.Linear(dims[-2], dims[-1], bias=False)

    def forward(self, x:torch.Tensor):
        h = self.net(x)
        logits = self.logit_module(h)
        dist = Categorical(logits=logits)
        return dist

class Encoder(nn.Module):
    def __init__(self, rnn_dims:list, y_dims:list, z_dims:list, ):#latent_dim:int, encoder_layer_dims:list,  cluster_dim:int,  z_input_dim:int):
        super().__init__()
        self.rnn_dims = rnn_dims
        self.y_dims = y_dims
        self.z_dims = z_dims
        self.rnn_hidden_dim = rnn_dims[-1]
        self.y_dim = y_dims[-1]
        self.y_module, self.rnn_module, self.z_module = self.create_inference_nets()

    def create_inference_nets(self):
        y_module = MLPCategorical(self.y_dims)

        rnn = nn.GRU(
            input_size=self.rnn_dims[0],
            hidden_size=self.rnn_dims[1],
        )
        z_module = MLPNormal(self.z_dims)
        return y_module, rnn, z_module

    def forward(self, xt, temp, mask):
        """
        Input:
            xt: Tensor of shape T x BS x (self.x_embedding_layer[-1]+self.t_embedding_layer[-1])
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
        last_seq = torch.argmax(mask , dim =0)#Time dimension - (BS,)
        # Pick out the state corresponding to last time step for each batch data point
        final_state = torch.cat([hidden_seq[last_seq[i],i][None, :] for i in range(BS)], dim = 0) #(BS,rnn_hidden_dim)
        try:
            assert final_state.shape == (BS, self.rnn_hidden_dim)
        except AssertionError:
            print(final_state.shape)
            raise
        
        dist_y = self.y_module(final_state)#[None, :, :] #shape(dist_y.logits) = (BS, y_dim)
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


class Model1(nn.Module):
    def __init__(self, latent_dim=20, marker_dim=31, marker_type='real', hidden_dim=128, time_dim=2, n_cluster=5, x_given_t=False, time_loss='normal', gamma=1., dropout=None, base_intensity=0, time_influence=0.1):
        super().__init__()
        self.marker_type = marker_type
        self.marker_dim = marker_dim
        self.time_dim = time_dim
        self.rnn_hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cluster_dim = n_cluster
        self.x_given_t = x_given_t
        self.time_loss = time_loss
        self.sigma_min = 1e-10
        self.gamma = gamma
        self.dropout = dropout

        # Preprocessing networks
        # Embedding network
        self.x_embedding_layer = [128]
        self.t_embedding_layer = [8]
        self.emb_dim = self.x_embedding_layer[-1] + self.t_embedding_layer[-1]
        self.embed_x, self.embed_t = self.create_embedding_nets()
        self.shared_output_layers = [256]
        self.inf_pre_module, self.gen_pre_module = self.create_preprocess_nets()

        # Forward RNN
        self.rnn = self.create_rnn()

        # Inference network
        self.encoder_rnn_hidden_dims = [64, 64]
        self.encoder_y_hidden_dims = [64]
        z_input_dim = self.rnn_hidden_dim + self.emb_dim + self.cluster_dim
        rnn_dims = [self.emb_dim, self.rnn_hidden_dim]
        y_dims = [self.rnn_hidden_dim, *self.encoder_y_hidden_dims, self.cluster_dim]
        z_dims = [z_input_dim, *self.encoder_rnn_hidden_dims, self.latent_dim]
        self.encoder = Encoder(rnn_dims=rnn_dims, y_dims=y_dims, z_dims=z_dims)

        # Generative network
        create_output_nets(self, base_intensity, time_influence)

    def create_embedding_nets(self):
        # marker_dim is passed. timeseries_dim is 2
        if self.marker_type == 'categorical':
            x_module = nn.Embedding(self.marker_dim, self.x_embedding_layer[0])
        else:
            raise NotImplementedError
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
        hxty_input_dim = self.rnn_hidden_dim+self.x_embedding_layer[-1]+self.t_embedding_layer[-1]+self.cluster_dim
        inf_pre_module = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(hxty_input_dim, hxty_input_dim),
            nn.ReLU(),nn.Dropout(self.dropout))

        # Generative net preprocessing
        hzy_input_dim = self.rnn_hidden_dim+self.latent_dim+self.cluster_dim
        gen_pre_module = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.Linear(hzy_input_dim, self.shared_output_layers[-1]),
            nn.ReLU(),nn.Dropout(self.dropout))
        return inf_pre_module, gen_pre_module


    def create_rnn(self):
        rnn = nn.GRU(
            input_size=self.x_embedding_layer[-1]+self.t_embedding_layer[-1],
            hidden_size=self.rnn_hidden_dim,
        )
        return rnn

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

    def forward(self, marker_seq, time_seq, anneal=1., mask=None, temp=0.5, preds_file=None):
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
        concat_hzy = torch.cat([hidden_seq[:-1], posterior_sample_z, posterior_sample_y], dim=-1)
        phi_hzy = self.gen_pre_module(concat_hzy)
        mu_marker, logvar_marker = generate_marker(self, phi_hzy, None)
        time_log_likelihood, mu_time = compute_point_log_likelihood(self, phi_hzy, t)
        marker_log_likelihood = compute_marker_log_likelihood(self, x, mu_marker, logvar_marker)

        KL_cluster = kl_divergence(posterior_dist_y, prior_dist_y)
        KL_z = kl_divergence(posterior_dist_z, prior_dist_z).sum(-1)*mask
        KL = KL_cluster.sum() + KL_z.sum()
        # try:
        #     assert (KL >= 0)
        # except:
        #     import pdb; pdb.set_trace()





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