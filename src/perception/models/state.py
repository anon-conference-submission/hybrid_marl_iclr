import torch as th
from perception.models.model import PerceptionModel


class State(PerceptionModel):

    def __init__(self, scheme, args):
        super(State, self).__init__(scheme,args)
        state_dim = scheme["state"]["vshape"]
        action_dim = args.n_actions
        self.data_dim = state_dim + action_dim
        self.state_dim = state_dim
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
        rl_input_dim = self.scheme["state"]["vshape"]
        return rl_input_dim

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):
        state_t = ep_batch["state"][:, t] # [batch_size,state_dim]
        state_t = state_t.unsqueeze(1) # [batch_size,1,state_dim]
        inputs = state_t.repeat(1, self.args.n_agents, 1)  # [batch_size,num_agents,state_dim]
        agent_inputs = inputs.unsqueeze(1) # [batch_size,1,num_agents,state_dim]
        return agent_inputs.detach()
