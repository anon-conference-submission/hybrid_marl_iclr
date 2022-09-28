import time
import numpy as np
from functools import partial

from envs import REGISTRY as env_REGISTRY
from components.episode_buffer import EpisodeBatch


class VisualizationRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.test_returns_comm_p = {}
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, perception_model=None, rl_scheme=None):
        # Batch to store in replay buffer.
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        # Batch containing data for action selection.
        self.new_action_selection_batch = partial(EpisodeBatch, rl_scheme, groups, self.batch_size,
                            self.episode_limit + 1, preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.perc_model = perception_model

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.action_selection_batch = self.new_action_selection_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, comm_p=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.perc_model:
            self.perc_model.init_perception_model(batch_size=self.batch_size)

        while not terminated:
            print('t=', self.t)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            if self.perc_model:
                perc_model_out = self.perc_model.encode(self.batch, t=self.t,
                            test_mode=test_mode, comm_p=comm_p) # [1,num_agents,latent_dim]
                pre_transition_data["obs"] = [[s for s in perc_model_out.numpy()[0]]]
            self.action_selection_batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.action_selection_batch, t_ep=self.t,
                                            t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.action_selection_batch.update(post_transition_data, ts=self.t)

            self.t += 1

            self.env.render()
            time.sleep(0.2)

        print('-'*20)
        print('Episode finished.')
        print("Episode return:", episode_return)
        input("Press any key to skip to the next episode.")

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        if self.perc_model:
            perc_model_out = self.perc_model.encode(self.batch, t=self.t,
                        test_mode=test_mode, comm_p=comm_p) # [1,num_agents,latent_dim]
            last_data["obs"] = [[s for s in perc_model_out.cpu().numpy()[0]]]
        self.action_selection_batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.action_selection_batch, t_ep=self.t,
                                            t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        self.action_selection_batch.update({"actions": actions}, ts=self.t)

        return self.batch

