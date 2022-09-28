import torch as th
from perception.models.model import PerceptionModel


class JointObsModel(PerceptionModel):

    def __init__(self, scheme, args):
        super(JointObsModel, self).__init__(scheme,args)
        obs_dim = scheme["obs"]["vshape"]
        action_dim = args.n_actions
        self.data_dim = obs_dim + action_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @property
    def is_trainable(self):
        return False

    @property
    def is_evaluated_with_different_comm_levels(self):
        return False

    def get_network(self):
        pass

    def init_perception_model(self, batch_size):
        pass

    def get_rl_input_dim(self):
        obs_dim = self.scheme["obs"]["vshape"]
        rl_input_dim = self.args.n_agents * obs_dim
        return rl_input_dim

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):
        agent_inputs = self._build_inputs(ep_batch, t) # [batch_size,1,num_agent,obs_dim]

        # Discard last action and concatenate the observations of all agents.
        agent_inputs = agent_inputs[:,:,:,:self.obs_dim]

        latent = agent_inputs.reshape(agent_inputs.shape[0],agent_inputs.shape[1],-1) # [batch_size,1,latent_dim]
        latent = latent.repeat(1, self.args.n_agents, 1)  # [batch_size,num_agents,latent_dim]

        return latent.detach()
