import torch as th


class PerceptionModel(object):
    """
        Generic perception model class.
    """

    def __init__(self, scheme, args):
        self.scheme = scheme
        self.args = args

    @property
    def is_trainable(self):
        raise NotImplementedError()

    @property
    def is_evaluated_with_different_comm_levels(self):
        raise NotImplementedError()

    def get_network(self):
        raise NotImplementedError()

    def init_perception_model(self, batch_size):
        raise NotImplementedError()

    def get_rl_input_dim(self):
        raise NotImplementedError()

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):
        raise NotImplementedError()

    def build_inputs(self, batch):
        # Used for training purposes.
        # Unpack dimensions.
        _, n_timesteps, _, _ = batch["obs"].shape
        n_actions = batch["actions_onehot"].shape[-1]

        inputs = []
        inputs.append(batch["obs"]) # [batch_size,num_timesteps,agents,obs_size]

        # Append last action to input (by default it is always appended).
        actions = th.ones_like(batch["actions_onehot"]) / n_actions # [batch_size,num_timesteps,agents,num_actions]
        for t in range(1, n_timesteps):
            actions[:,t,:,:] = batch["actions_onehot"][:, t-1]
        inputs.append(actions)

        inputs = th.cat(inputs, dim=-1) # [batch_size,num_timesteps,agents,obs_dim]

        return inputs

    def _build_inputs(self, batch, t):
        # Used for encoding purposes.
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        n_actions = batch["actions_onehot"].shape[-1]

        inputs = []
        inputs.append(batch["obs"][:, t])

        # Append last action to input (by default it is always appended).
        if t == 0:
            inputs.append(th.ones_like(batch["actions_onehot"][:, t]) / n_actions)
        else:
            inputs.append(batch["actions_onehot"][:, t-1])

        inputs = th.cat(inputs, dim=-1)
        inputs = inputs.reshape(bs,1, self.args.n_agents,-1) # [batch_size,1,num_agents,obs_dim]

        return inputs

    def process_batch(self, ep_batch, test_mode=False):
        self.init_perception_model(batch_size=ep_batch.batch_size)
        outs = []
        for t in range(ep_batch.max_seq_length):
            outs.append(self.encode(ep_batch, t=t))
        outs = th.stack(outs, dim=1)
        ep_batch.data.transition_data["obs"] = outs
        return ep_batch
