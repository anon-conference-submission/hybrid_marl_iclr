"""
Define the neural network modules used by the perception models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal


# Test networks
class TestEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TestEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc1(x)

class TestDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TestDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        return out, out

# Linear Networks
class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.m_z = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.m_z(h)

class LinearGaussianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearGaussianEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

class LinearDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_obs_dim, output_act_dim):
        super(LinearDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_obs_dim)
        self.fc4 = nn.Linear(hidden_dim, output_act_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        out_obs = self.fc3(h)
        out_act = F.softmax(self.fc4(h), dim=-1)
        return out_obs, out_act

# World model autoencoder.
class HwmAETest(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim, obs_dim, act_dim):
        super(HwmAETest, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoder = TestEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TestDecoder(latent_dim, hidden_dim, input_dim)

    def training_loss(self, recon_obs, recon_act, x, mask):

        # Observation reconstruction loss
        return torch.nn.functional.mse_loss(input=recon_obs, target=x, reduction='none').sum(-1).sum(-1).sum(-1)

    def encode(self, x, sample=False):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        recon_obs, recon_act = self.decoder(z)
        return recon_obs, recon_act, z


class HwmAE(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim, obs_dim, act_dim):
        super(HwmAE, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoder = LinearEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LinearDecoder(latent_dim, hidden_dim, obs_dim, act_dim)

    def training_loss(self, recon_obs, recon_act, x, mask):

        # Extract agent-specific data
        x_obs = x[:, :, :, :self.obs_dim].clone()
        x_action = x[:, :, :, self.obs_dim:self.obs_dim + self.act_dim].clone()

        # Observation reconstruction loss
        recon_observation_loss = torch.nn.functional.mse_loss(input=recon_obs, target=x_obs, reduction='none')
        recon_obs_mask = mask.unsqueeze(-1).expand_as(recon_observation_loss).detach()
        recon_observation_loss = (recon_obs_mask * recon_observation_loss).sum(-1).sum(-1).sum(-1) / recon_obs_mask.sum(-1).sum(
            -1).sum(-1)

        # # Action reconstruction loss
        x_action = torch.argmax(x_action, dim=-1)  # NLL Loss requires target of type (Bsize, k_1, k_2, ..., k_n)

        out_act = torch.log(recon_act + 1E-10)  # NLL loss requires Log Softmax
        out_act = out_act.permute(0, 3, 1, 2).contiguous()  # NLL Loss requires input of type (Bsize, #Classes, k_1, k_2, ..., k_n)

        recon_action_loss = torch.nn.functional.nll_loss(input=out_act, target=x_action,
                                                         reduction='none')
        recon_action_mask = mask.expand_as(recon_action_loss).detach()
        recon_action_loss = recon_action_loss * recon_action_mask
        recon_action_loss = recon_action_loss.sum(-1).sum(-1)/ recon_action_mask.sum(-1).sum(-1)

        return recon_observation_loss + recon_action_loss

    def encode(self, x, sample=False):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        recon_obs, recon_act = self.decoder(z)
        return recon_obs, recon_act, z


# World model variational autoencoder.
class HwmVAE(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dim, obs_dim, act_dim):
        super(HwmVAE, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.encoder = LinearGaussianEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LinearDecoder(latent_dim, hidden_dim, obs_dim, act_dim)

    def reparametrize(self, mu, logvar):

        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()

        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)

    def training_loss(self, recon_obs, recon_act, x, mu, logvar, mask):
        #recon_loss = F.mse_loss(recon, x, size_average=False)

        # Extract agent-specific data
        x_obs = x[:, :, :, :self.obs_dim]
        x_action = x[:, :, :, self.obs_dim:self.obs_dim + self.act_dim]

        # Observation reconstruction loss
        recon_observation_loss = torch.nn.functional.mse_loss(input=recon_obs, target=x_obs, reduction='none')
        recon_obs_mask = mask.unsqueeze(-1).expand_as(recon_observation_loss).detach()
        recon_observation_loss = (recon_obs_mask * recon_observation_loss).sum(-1).sum(-1).sum(-1) / recon_obs_mask.sum(
            -1).sum(
            -1).sum(-1)

        # Action reconstruction loss
        x_action = torch.argmax(x_action, dim=-1)  # NLL Loss requires target of type (Bsize, k_1, k_2, ..., k_n)

        out_act = torch.log(recon_act + 1E-10)  # NLL loss requires Log Softmax
        out_act = out_act.permute((0, 3, 1, 2)).contiguous()  # NLL Loss requires input of type (Bsize, #Classes, k_1, k_2, ..., k_n)

        recon_action_loss = torch.nn.functional.nll_loss(input=out_act, target=x_action,
                                                         reduction='none')
        recon_action_mask = mask.expand_as(recon_action_loss).detach()
        recon_action_loss = recon_action_loss * recon_action_mask
        recon_action_loss = recon_action_loss.sum(-1).sum(-1) / recon_action_mask.sum(-1).sum(-1)

        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp(), dim=-1)
        kld_mask = mask.expand_as(KLD).detach()
        KLD = (kld_mask*KLD).sum(-1).sum(-1) / kld_mask.sum(-1).sum(-1)

        return recon_observation_loss + recon_action_loss + KLD

    def encode(self, x, sample=False):
        mu, logvar = self.encoder(x)
        if sample:
            return self.reparametrize(mu, logvar)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        recon_obs, recon_act = self.decoder(z)
        return recon_obs, recon_act, z, mu, logvar


# Multi-dimensional RNN.
class MDRNN(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_agents, n_gaussians):
        super(MDRNN, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.n_agents = n_agents
        self.lstm = nn.LSTM(latent_dim*n_agents, hidden_dim, batch_first=True)
        self.gmm_linear = nn.Linear(hidden_dim, (2 * latent_dim + 1) * n_gaussians * n_agents)


    def encode(self, latents, hidden):

        """ ONLY FOR TESTING - SINGLE Step forward.
            :args latents: [BSIZE, LSIZE * N_AGENTS] torch tensor

            :returns: mu_nlat, sig_nlat, pi_nlat, parameters of the GMM
            prediction for the next latent.

                - mu_nlat: (BSIZE, N_AGENTS, N_GAUSS, LSIZE) torch tensor
                - sigma_nlat: (BSIZE, N_AGENTS, N_GAUSS, LSIZE) torch tensor
                - logpi_nlat: (BSIZE, N_AGENTS, N_GAUSS) torch tensor
        """

        batchsize = latents.shape[0]
        in_latents = latents.unsqueeze(1)  # Shape = [batchsize, 1, l_dim*n_agents]

        outs, hidden = self.lstm(in_latents, hidden)
        gmm_outs = self.gmm_linear(outs) # SHAPE = ??

        stride = self.n_gaussians * self.latent_dim * self.n_agents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(batchsize, self.n_agents, self.n_gaussians, self.latent_dim)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(batchsize, self.n_agents, self.n_gaussians, self.latent_dim)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.n_gaussians * self.n_agents]
        pi = pi.view(batchsize, self.n_agents, self.n_gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        return mus, sigmas, logpi, hidden

    def training_loss(self, x, mus, sigmas, logpi, mask, reduce=True):

        """ Computes the gmm loss.

        Compute minus the log probability of batch under the GMM model described
        by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
        dimensions (several batch dimension are useful when you have both a batch
        axis and a time step axis), gs the number of mixtures and fs the number of
        features.

        :args batch: (bs1, bs2, *, fs) torch tensor - NEXT TIME STEP
        :args mus: (bs1, bs2, *, gs, fs) torch tensor
        :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
        :args logpi: (bs1, bs2, *, gs) torch tensor
        :args reduce: if not reduce, the mean in the following formula is ommited

        :returns:
        loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
            sum_{k=1..gs} pi[i1, i2, ..., k] * N(
                batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

        NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
        with fs).
        """
        batch = x.unsqueeze(-2)
        normal_dist = Normal(mus, sigmas)
        g_log_probs = normal_dist.log_prob(batch)
        g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
        max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
        g_log_probs = g_log_probs - max_log_probs

        g_probs = torch.exp(g_log_probs)
        probs = torch.sum(g_probs, dim=-1)

        log_prob = max_log_probs.squeeze() + torch.log(probs)

        prob_mask = mask[:, :-1, :].expand_as(log_prob).detach()
        return - (prob_mask * log_prob).sum(-1).sum(-1)/ prob_mask.sum(-1).sum(-1)

        #if reduce:
        #    return - torch.mean(log_prob)
        #return -log_prob


    def forward(self, latents):

        """ ONLY FOR TRAINING - MULTI steps forward.
        :args latents: [BSIZE, SEQ_LEN, LSIZE* N_AGENTS] torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, parameters of the GMM
        prediction for the next latent.

            - mu_nlat: (BSIZE, SEQ_LEN, N_AGENTS, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, SEQ_LEN, N_AGENTS, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, SEQ_LEN, N_AGENTS, N_GAUSS) torch tensor
        """

        batchsize, seq_len = latents.shape[0], latents.shape[1]

        outs, hidden = self.lstm(latents)
        gmm_outs = self.gmm_linear(outs)  # SHAPE = ??

        stride = self.n_gaussians * self.latent_dim * self.n_agents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(batchsize, seq_len, self.n_agents, self.n_gaussians, self.latent_dim)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(batchsize, seq_len, self.n_agents, self.n_gaussians, self.latent_dim)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.n_gaussians * self.n_agents]
        pi = pi.view(batchsize, seq_len, self.n_agents, self.n_gaussians)
        logpi = F.log_softmax(pi, dim=-1)

        return mus, sigmas, logpi
