import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_landmarks = 1
        world.collaborative = True

        # add agents
        # Speaker agent
        speaker_agent = Agent()
        speaker_agent.name = "speaker"
        speaker_agent.collide = False
        speaker_agent.size = 0.075
        speaker_agent.movable = True
        speaker_agent.silent = True
        speaker_agent.color = np.array([0.25,0.25,0.25])

        # Listener agent
        listener_agent = Agent()
        listener_agent.name = "listener"
        listener_agent.collide = False
        listener_agent.size = 0.075
        listener_agent.movable = True
        listener_agent.silent = True
        listener_agent.color = np.array([0.45,0.45,0.25])

        world.agents = [speaker_agent, listener_agent]

        # add goal
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None

        # want the speaker and listener to both go to the goal landmark
        world.agents[0].goal_b = world.np_random.choice(world.landmarks)
        world.agents[1].goal_b = world.agents[0].goal_b

        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.color = np.array([0.15, 0.15, 0.15])

        # special colors for goals
        world.agents[0].goal_b.color = np.array([0.85, 0.85, 0.85])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.np_random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = world.np_random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return self.reward(agent, world)

    def reward(self, agent, world):

        # Speaker distance
        speak_dist2 = np.sum(np.square(world.agents[0].state.p_pos - world.agents[0].goal_b.state.p_pos))

        # Listener distance
        list_dist2 = np.sum(np.square(world.agents[1].state.p_pos - world.agents[0].goal_b.state.p_pos))

        return -np.mean([list_dist2, speak_dist2])

    def observation(self, agent, world):

        if agent.name == 'speaker':
            # get positions of goal
            return np.concatenate([world.agents[0].goal_b.state.p_pos])

        else:
            agents_pos, agents_vel = [], []
            for ag in world.agents:
                agents_pos.append(ag.state.p_pos)
                agents_vel.append(ag.state.p_vel)

            return np.concatenate(agents_pos + agents_vel)