import torch as th
import numpy as np

from perception.models.model import PerceptionModel
from perception.models.sampling_schemes import sampling_registry


class MaskedJointObsModel(PerceptionModel):

    def __init__(self, scheme, args):
        super(MaskedJointObsModel, self).__init__(scheme,args)
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
        return True

    def get_network(self):
        pass

    def init_perception_model(self, batch_size):
        if isinstance(self.args.perception_args["train_comm_p"], float):
            self.train_comm_p = self.args.perception_args["train_comm_p"]
        elif isinstance(self.args.perception_args["train_comm_p"], str):
            self.train_comm_p = sampling_registry[self.args.perception_args["train_comm_p"]]()
        else:
            raise ValueError("Incorrect sampling scheme selected:" + str(self.args.perception_args["train_comm_p"]))

    def get_rl_input_dim(self):
        obs_dim = self.scheme["obs"]["vshape"]
        rl_input_dim = self.args.n_agents * obs_dim # joint_obs
        if self.args.perception_args["append_masks_to_rl_input"]:
            rl_input_dim += self.args.n_agents # array encoding the which observations are valid.
        return rl_input_dim

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):

        if not test_mode:
            # Train mode.
            comm_p = self.train_comm_p

        if self.args.perception_args["comm_at_t0"] and t == 0:
            comm_p = 1.0

        agent_inputs = self._build_inputs(ep_batch, t) # [batch_size,1,num_agent,obs_dim]

        # Discard last action and concatenate the observations of all agents.
        agent_inputs = agent_inputs[:,:,:,:self.obs_dim] # [batch_size,1,num_agent,obs_dim]

        latent = agent_inputs.reshape(agent_inputs.shape[0],agent_inputs.shape[1],-1) # [batch_size,1,obs_dim*num_agents]
        latent = latent.repeat(1, self.args.n_agents, 1)  # [batch_size,num_agents,obs_dim*num_agents]

        # Generate mask given `comm_p` probability.
        n_agents = latent.shape[1]
        bsize = latent.shape[0]
        agent_comm_mask = th.rand(bsize,n_agents,n_agents) # [batch_size,num_agents,num_agents]
        agent_comm_mask = agent_comm_mask + th.eye(n_agents) #.repeat(bsize,1,1) (agent always communicates with itself).
        agent_comm_mask = (agent_comm_mask >= (1.0 - comm_p)) # Mask out entries.
        agent_comm_mask_repeated = th.repeat_interleave(agent_comm_mask, self.obs_dim, axis=-1)

        # Mask the entries of the latent vector using the agent_comm_mask_repeated variable.
        latent = th.where(agent_comm_mask_repeated, latent, th.zeros_like(latent))

        # Append to the observations an array encoding which entries are valid.
        if self.args.perception_args["append_masks_to_rl_input"]:
            latent = th.concat([latent,agent_comm_mask.type(th.float32)], dim=-1)

        return latent.detach()
