1) Copy python files in this folder to the folder with the mpe fork from Epymarl - `multiagent-particle-envs/mpe/scenarios/`

2) In the `__init__.py` file of the folder `multiagent-particle-envs/mpe`, register the scenarios:
```

_particles = {
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_blind_deaf": "SimpleBlindDeaf-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_spread_blind": "SimpleSpreadBlind-v0",
    "simple_spread_xy": "SimpleSpreadXY-v0",
    "simple_spread_xy_4": "SimpleSpreadXY4-v0",
    "simple_spread_xy_8": "SimpleSpreadXY8-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
    "climbing_spread": "ClimbingSpread-v0",
}

```


3) Run in the top folder `multiagent-particle-envs`, the command to install the package:

```
pip install -e .
```

4) Run Epymarl as usual (e.g.):

```

python src/main.py --config=iql_ns --env-config=gymma with env_args.key="SimpleSpreadBlind-v0"

python src/main.py --config=iql_ns --env-config=gymma with env_args.key="SimpleSpreadMyopic01-v0"

python src/main.py --config=iql_ns --env-config=gymma with env_args.key="SimpleBlindDeaf-v0"


```