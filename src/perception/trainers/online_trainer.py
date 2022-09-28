from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam


class OnlineTrainer:
    def __init__(self, perc_model, logger, args):

        # Internalise arguments.
        self.args = args
        self.logger = logger
        self.perc_model = perc_model
        self.network = perc_model.get_network()

        # Optimizer.
        self.optim = Adam(self.network.parameters(),
                    lr=self.args.perception_args['learning_rate'])

        self.log_stats_t = -self.args.perception_args['trainer_log_interval'] - 1

    def train(self, batch: EpisodeBatch, t_env: int):
        obs = self.perc_model.build_inputs(batch) # [batch_size, num_timesteps, agents, obs_dim]
        obs = obs[:, :-1] # [batch_size, num_timesteps-1, agents, obs_dim]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        train_params = {"parameter": None}
        loss, loss_info = self.network.training_step(obs, mask, train_params)

        loss.backward()
        if self.args.perception_args["grad_clip"]:
            th.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.perception_args["grad_clip"])
        self.optim.step()

        # Log losses.
        if t_env - self.log_stats_t >= self.args.perception_args['trainer_log_interval']:
            for key, val in loss_info.items():
                self.logger.log_stat("perc_" + key, val, t_env)
            self.log_stats_t = t_env

    def save_models(self, path, save_mongo=False):
        th.save(self.network.state_dict(), "{}/network.th".format(path))
        th.save(self.optim.state_dict(), "{}/opt.th".format(path))

        if save_mongo:
            self.logger.log_model(filepath="{}/network.th".format(path), name="perceptual_model.th")
            self.logger.log_model(filepath="{}/opt.th".format(path), name="perceptual_opt.th")

    def load_models(self, path):
        if self.args.use_cuda:
            checkpoint = th.load("{}/network.th".format(path))
            checkpoint_opt = th.load("{}/opt.th".format(path))
        else:
            checkpoint = th.load("{}/network.th".format(path),
                map_location=lambda storage, location: storage)
            checkpoint_opt = th.load("{}/opt.th".format(path),
                map_location=lambda storage, location: storage)
        self.network.load_state_dict(checkpoint)
        self.optim.load_state_dict(checkpoint_opt)
