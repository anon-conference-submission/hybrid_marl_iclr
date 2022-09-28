import copy
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions.categorical import Categorical

from perception.models import nets
from perception.models.model import PerceptionModel
from perception.models.sampling_schemes import sampling_registry




class HybridMDRNN(PerceptionModel):

    def __init__(self, scheme, args):
        super(HybridMDRNN, self).__init__(scheme,args)

        # Instantiate network.
        self.network = HybridMDRNNNetwork(
            n_agents=args.n_agents,
            obs_dim=scheme["obs"]["vshape"],
            act_dim=args.n_actions,
            hidden_dim=args.perception_args["hidden_dim"],
            n_gaussians=args.perception_args["n_gaussians"],
        )

        self.hidden_state_agents = None
        self.estimated_agents_obs = None

        if args.use_cuda:
            self.network.cuda()

    @property
    def is_trainable(self):
        return True

    @property
    def is_evaluated_with_different_comm_levels(self):
        return True

    def get_network(self):
       return self.network

    def init_perception_model(self, batch_size=1):
        if not self.args.use_cuda:
            self.hidden_state_agents = [ (th.zeros((1, batch_size, self.network.hidden_dim)),
                                        th.zeros((1, batch_size, self.network.hidden_dim)))
                                        for _ in range(self.args.n_agents) ]
        else:
            self.hidden_state_agents = [(th.zeros((1, batch_size, self.network.hidden_dim)).cuda(),
                                         th.zeros((1, batch_size, self.network.hidden_dim)).cuda())
                                        for _ in range(self.args.n_agents)]

        self.estimated_agents_obs = [[None] * self.args.n_agents
                                        for _ in range(self.args.n_agents)]

        if isinstance(self.args.perception_args["train_comm_p"], float):
            self.train_comm_p = self.args.perception_args["train_comm_p"]
        elif isinstance(self.args.perception_args["train_comm_p"], str):
            self.train_comm_p = sampling_registry[self.args.perception_args["train_comm_p"]]()
        else:
            raise ValueError("Incorrect sampling scheme selected:" + str(self.args.perception_args["train_comm_p"]))

    def get_rl_input_dim(self):
        obs_dim = self.scheme["obs"]["vshape"]
        act_dim = self.args.n_actions

        rl_input_dim = (obs_dim + act_dim) * self.args.n_agents

        if self.args.perception_args["append_masks_to_rl_input"]:
            rl_input_dim +=  self.args.n_agents

        return rl_input_dim

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):
        """
            Encodes the observations. Can be called at:
                (i) train time for action selection (batch size can be greater than 1 if
                    multiple envs are running in parallel);
                (ii) test time for action selection (batch size can be greater than 1 if
                    multiple envs are running in parallel);
                (iii) train time to process the batch sampled from the replay buffer
                    before passing it to the RL module (batch size equals the replay
                    buffer batch size).
        """
        if not test_mode:
            # Train mode.
            comm_p = self.train_comm_p

        if self.args.perception_args["comm_at_t0"] and t == 0:
            comm_p = 1.0

        agent_inputs = self._build_inputs(ep_batch, t) # [batch_size,1,num_agent,data_dim]

        latents = []
        for agent_id in range(self.args.n_agents):

            # Generate random agent id list given `comm_p` probability.
            agent_com = np.random.choice([0, 1], size=self.args.n_agents, p=[(1 - comm_p), comm_p])
            agent_com[agent_id] = 1  # Agent always communicates with itself.
            agent_mask = agent_com
            agent_com = np.where(agent_com == 1)[0].tolist()

            latent_agent, \
            self.hidden_state_agents[agent_id], \
            self.estimated_agents_obs[agent_id] = self.network.agent_encode(agent_inputs,
                                                    hidden=self.hidden_state_agents[agent_id],
                                                    agent_estimated_obs=self.estimated_agents_obs[agent_id],
                                                    agent_id=agent_id,
                                                    agent_com=agent_com) # [batch_size, data_dim+hidden_dim]

            if self.args.perception_args["append_masks_to_rl_input"]:
                tensor_comm_mask = th.FloatTensor(agent_mask).unsqueeze(0)
                tensor_comm_mask = th.repeat_interleave(tensor_comm_mask, agent_inputs.shape[0], axis=0).to(agent_inputs.device)
                latent_agent = th.cat([latent_agent, tensor_comm_mask], dim=-1)  # [batch_size, data_dim*n_agents + n_agents]

            latents.append(latent_agent.unsqueeze(1))  # [batch_size,1,data_dim+hidden_dim (+n_agents)]

        latent = th.concat(latents, dim=1)  # [batch_size,num_agents,data_dim+hidden_dim (+n_agents)]
        return latent.detach()


class HybridMDRNNNetwork(nn.Module):

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim, n_gaussians):
        super(HybridMDRNNNetwork, self).__init__()

        self.n_agents = n_agents
        self.data_dim = obs_dim + act_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians

        # Forward prediction model (P).
        self.prediction_model = nets.MDRNN(latent_dim=self.data_dim, hidden_dim=hidden_dim,
                        n_gaussians=n_gaussians, n_agents=n_agents)

    def agent_encode(self, data, hidden=None, agent_estimated_obs=None,
                        agent_id=None, agent_com=None):

        # data shape = [batch_size,1,num_agent,data_dim]

        latents = []
        for i_ag in range(self.n_agents):

            # If we communicate the observations
            if i_ag in agent_com:

                agent_obs = data[:, :, i_ag, :].squeeze(1)                          # Squeeze out timestep dim - [bsize, data_dim]
                latents.append(agent_obs)                                           # [bsize, data_dim]
            else:
                # If we don't communicate the observations
                # If we don't have any estimate (only for first timestep)
                if agent_estimated_obs[i_ag] is None:
                    latents.append(th.zeros((data.shape[0], self.data_dim),
                                               device=data.device))                 # [bsize, data_dim]

                # If we have an estimate (from a previous timestep)
                else:
                    latents.append(agent_estimated_obs[i_ag])

        # Get data for RL input
        rl_input_data = th.cat(latents, dim=-1)                                        # [bsize, data_dim*#agents]

        # Update Predictive model
        mus, sigmas, logpi, hidden = self.prediction_model.encode(rl_input_data, hidden)

        # Update estimates of agent latent states - We take only the mean of the sampled gaussian from the MoG
        next_agent_estimated_obs = [None] * self.n_agents
        for i_ag in range(self.n_agents):

            # If we use a mixture of gaussians model (K > 1)
            if self.n_gaussians != 1:

                # Sample one mixture
                ag_mixt = Categorical(th.exp(logpi[:, i_ag, :].squeeze())).sample() # shape = [bsize, 1]

                # Update the agent-specific state estimates
                next_agent_estimated_obs[i_ag] = mus[:, i_ag, ag_mixt, :]    # shape = [bsize, latent_dim]

            # If we are using a single gaussian (K = 1)
            else:
                next_agent_estimated_obs[i_ag] = mus[:, i_ag, 0, :]  # shape = [bsize, latent_dim]

        return rl_input_data.detach(), hidden, next_agent_estimated_obs

    def forward(self, x):

        # x - shape [batch-size, #timesteps, #agents, obs_dim]

        # Encode latents with the predictive model
        in_latents = x.view(x.shape[0],x.shape[1],-1)          # shape = [bsize, #timesteps, latent_dim*#agents]

        mus, sigmas, logpi = self.prediction_model(in_latents.clone().detach())

        # Drop last element of the prediction model output
        mus    = mus[:, :-1, :, :, :]                  # shape = [batch-size, #timesteps - 1, #agents, #Gaussians, latent_dim]
        sigmas = sigmas[:, :-1, :, :, :]               # shape = [batch-size, #timesteps - 1, #agents, #Gaussians, latent_dim]
        logpi  = logpi[:, :-1, :, :]                   # shape = [batch-size, #timesteps - 1, #agents, #Gaussians]

        return mus, sigmas, logpi, x

    def training_step(self, data, mask, train_params):

        loss_info = {}

        # Forward Pass
        pred_mus, pred_sigmas, pred_logpi, out_latents, = self.forward(data)

        # World Model Loss
        next_latents = th.roll(out_latents.detach(), -1,  dims=1)     # Roll and drop last element.
        next_latents = next_latents[:, :-1, :, :].clone().detach()
        total_wm_loss = self.prediction_model.training_loss(x=next_latents,
                                                            mus=pred_mus,
                                                            sigmas=pred_sigmas,
                                                            logpi=pred_logpi,
                                                            mask=mask,
                                                            reduce=True)

        loss_info['predictor_loss'] = th.mean(total_wm_loss).cpu().item()

        # Compute total_loss
        total_loss = th.mean(total_wm_loss)
        loss_info['total_loss'] = th.mean(total_loss).cpu().item()

        return total_loss, loss_info
