import datetime
import os
import pprint
import time
import threading
import numpy as np
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from perception.models import REGISTRY as perc_model_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from perception.trainers.online_trainer import OnlineTrainer

from tqdm import tqdm

COMMS_VEC = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 'unif']


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]   
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
    logger.console_logger.info("Experiment token ID: {}".format(unique_token))

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner, perc_model=None):

    if perc_model and \
        perc_model.is_evaluated_with_different_comm_levels:
        for comm_p in COMMS_VEC:
            for _ in range(args.test_nepisode):
                runner.run(test_mode=True, comm_p=comm_p)
    else:
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup perception layer.
    perc_model = None
    if args.perception_args["perception"]:

        # Build perception model.
        perc_model = perc_model_REGISTRY[args.perception_args["model_type"]](scheme, args)

        # Setup trainer.
        if perc_model.is_trainable:
            perc_trainer = OnlineTrainer(perc_model=perc_model, logger=logger, args=args)

        # Load perception model checkpoint.
        if perc_model.is_trainable and args.perception_args["checkpoint_path"] != "":

            timesteps = []
            timestep_to_load = 0

            if not os.path.isdir(args.perception_args["checkpoint_path"]):
                logger.console_logger.info(
                    "Checkpoint directory {} doesn't exist".format(args.perception_args["checkpoint_path"])
                )
                return

            # Go through all files in args.perception_args["checkpoint_path"]
            for name in os.listdir(args.perception_args["checkpoint_path"]):
                full_name = os.path.join(args.perception_args["checkpoint_path"], name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if args.perception_args["load_step"] == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - args.perception_args["load_step"]))
                
            model_path = os.path.join(args.perception_args["checkpoint_path"], str(timestep_to_load))
            logger.console_logger.info("Loading perception model from {}".format(model_path))
            
            perc_trainer.load_models(model_path)

    rl_scheme = scheme.copy()
    if args.perception_args["perception"]:
        rl_scheme["obs"] = {"vshape": perc_model.get_rl_input_dim(), "group": "agents"}

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](rl_scheme, groups, args)

    # Give runner the scheme.
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess,
                mac=mac, perception_model=perc_model, rl_scheme=rl_scheme)

    # Setup RL learner.
    learner = le_REGISTRY[args.learner](mac, rl_scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    # Load RL agent checkpoint.
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner, perc_model=perc_model)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    perc_model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    pbar = tqdm(total=args.t_max)

    while runner.t_env <= args.t_max:

        # Update progress bar.
        pbar.update(runner.t_env - pbar.n)

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            if args.perception_args["perception"]:

                # Train perception model.
                if perc_model.is_trainable:
                    perc_trainer.train(episode_sample, runner.t_env)
    
                # Process batch through the perception model.
                episode_sample = perc_model.process_batch(episode_sample, test_mode=False)

            # Train RL agent.
            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env

            if args.perception_args["perception"] and \
                perc_model.is_evaluated_with_different_comm_levels:

                for comm_p in COMMS_VEC:
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True, comm_p=comm_p)
            else:
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0 or  runner.t_env == args.t_max
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            if runner.t_env == args.t_max:
                learner.save_models(save_path, save_mongo=True)
            else:
                learner.save_models(save_path)

        if args.perception_args["perception"] and perc_model.is_trainable and \
            args.perception_args["save_model"] and (
            runner.t_env - perc_model_save_time >= args.perception_args["save_model_interval"]
            or perc_model_save_time == 0 or runner.t_env == args.t_max
        ):
            perc_model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "perception_models", args.unique_token, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving perception models to {}".format(save_path))

            # Save on Mongo the last perceptual model
            if runner.t_env == args.t_max:
                perc_trainer.save_models(save_path, save_mongo=True)
            else:
                perc_trainer.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
