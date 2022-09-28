# Hybrid MARL - Submission #2702

Hybrid MARL is an extension over the EPyMARL library.


# Installation & Run instructions

To install the codebase, clone this repo and install the `requirements.txt`.  
```sh
pip install -r requirements.txt
```

It is recomended to create a virtual environment with `python=3.8.10` before installing the requirements.

## Installing MPE standard and custom environments

For the MPE environments install:
- [The EpyMARL fork of MPE](https://github.com/semitable/multiagent-particle-envs), clone it in the main folder of our repository and install it by running `pip install -e .` inside the newly created folder `multiagent-particle-envs`.

After the installation, to setup our novel MPE environments, please follow the instructions in the `README.md` file available in the `src/envs/extra_envs` folder in the main folder of our repository.
Finally, install the EpyMARL package once again by running `pip install -e .` in the cloned `multiagent-particle-envs` folder.


## Running experiments
After installing them you can run it using (for example):
```sh
python src/main.py --config=iql_ns --env-config=gymma with env_args.key="SimpleSpreadXY-v0" seed="0"
```

where:
1) `--config=iql_ns` selects the RL algorithm to use (`iql_ns` or `qmix_ns`)
2) `env_args.key="SimpleSpreadXY-v0"` selects the environment to use:
    - `SimpleSpreadXY-v0` - Spread XY;
    - `SimpleSpreadBlind-v0` - Spread Blindfold;
    - `SimpleBlindDeaf-v0` - HearSee;
    - `SimpleSpread-v0` - Simple Spread;
    - `SimpleSpeakerListener-v0` - Simple Speaker Listener.
    
To change the agent type between MARO and the other baselines, modify the file `src/config/perception.yaml` accordingly to:
1) Observation (Baseline):
    - `perception=False`.
2) Joint Observation (Baseline):
    - `perception: True`;
    - `model: 'joint_obs'`;
    - `append_masks_to_rl_input: False`;
    - `p: 1.0`;
3) Message-Dropout (Baseline):
    - `perception: True`;
    - `model: 'masked_joint_obs'`;
    - `append_masks_to_rl_input: False`;
    - `p: 0.5`;   
4) MARO:
    - `perception: True`;
    - `model: 'mdrnn'`;
    - `append_masks_to_rl_input: True`;
    - `p: 'uniform'`;
 

