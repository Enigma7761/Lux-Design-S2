from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]

        for agent in obs.keys():
            factories = np.zeros(env_cfg.map_size)
            light_robots = np.zeros(env_cfg.map_size)
            heavy_robots = np.zeros(env_cfg.map_size)
            power = np.zeros(env_cfg.map_size)
            cargo_ice = np.zeros(env_cfg.map_size)
            cargo_ore = np.zeros(env_cfg.map_size)
            cargo_water = np.zeros(env_cfg.map_size)
            cargo_metal = np.zeros(env_cfg.map_size)

            for unit_id, unit in shared_obs['units'][agent].items():
                if unit.unit_type == "LIGHT":
                    light_robots[unit.pos] = 1
                else:
                    heavy_robots[unit.pos] = 1
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                power[unit.pos] = unit["power"] / battery_cap,
                cargo_ice[unit.pos] = unit["cargo"]["ice"] / cargo_space,
                cargo_ore[unit.pos] = unit["cargo"]["ore"] / cargo_space,
                cargo_water[unit.pos] = unit["cargo"]["water"] / cargo_space,
                cargo_metal[unit.pos] = unit["cargo"]["metal"] / cargo_space,
            for factory in shared_obs['factories'][agent]:
                factories[factory.pos] = 1

            board = shared_obs['board']
            rubble = board['rubble']
            ore = board['ore']
            ice = board['ice']
            lichen = board['lichen']

            observation[agent] = np.stack([factories, light_robots, heavy_robots, power,
                                           cargo_ice, cargo_ore, cargo_water, cargo_metal, rubble, ore, ice, lichen])

        return observation


class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Adds a custom reward and turns the LuxAI_S2 environment into a single-agent environment for easy training
        """
        super().__init__(env)
        self.prev_step_metrics = {'player_0': dict(), 'player_1': dict()}
        self.prev_obs = None

    def step(self, action):
        # TODO
        # submit actions for just one agent to make it single-agent
        # and save single-agent versions of the data below
        obs, _, done, info = self.env.step(action)

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        for agent in ['player_0', 'player_1']:
            stats = self.env.state.stats[agent]

            info = {'player_0': dict(), 'player_1': dict()}
            metrics = dict()
            metrics["ice_dug"] = (
                    stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"]
            )
            metrics["water_produced"] = stats["generation"]["water"]

            # we save these two to see often the agent updates robot action queues and how often enough
            # power to do so and succeed (less frequent updates = more power is saved)
            metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
            metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

            # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our
            # agent is behaving
            info[agent]["metrics"] = metrics

            reward = 0
            if self.prev_step_metrics[agent] is not None:
                # we check how much ice and water is produced and reward the agent for generating both
                ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics[agent]["ice_dug"]
                water_produced_this_step = (
                        metrics["water_produced"] - self.prev_step_metrics["water_produced"]
                )
                # we reward water production more as it is the most important resource for survival
                reward = ice_dug_this_step / 100 + water_produced_this_step

            self.prev_step_metrics[agent] = copy.deepcopy(metrics)
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        action = dict()

        for agent in self.env.agents:
            action[agent] = zero_bid(agent, obs[agent])
        obs, _, _, _ = self.env.step(action)

        # while real_env_steps < 0, we are in the factory placement phase
        # so we use the factory placement policy to step through this
        while self.env.state.real_env_steps < 0:
            action = dict()
            for agent in self.env.agents:
                if my_turn_to_place_factory(obs[agent]['teams'][agent]['place_first'], self.env.state.env_steps):
                    action[agent] = place_near_random_ice(agent, obs[agent])
                else:
                    action[agent] = dict()
            obs, _, _, _ = self.env.step(action)
        self.prev_obs = obs
        self.prev_step_metrics = {'player_0': dict(), 'player_1': dict()}

        return obs


def zero_bid(player, obs):
    # a policy that always bids 0
    faction = "AlphaStrike"
    if player == "player_1":
        faction = "MotherMars"
    return dict(bid=0, faction=faction)


def place_near_random_ice(player, obs):
    """
    This policy will place a single factory with all the starting resources
    near a random ice tile
    """
    if obs["teams"][player]["metal"] == 0:
        return dict()
    potential_spawns = list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
    potential_spawns_set = set(potential_spawns)
    done_search = False

    # simple numpy trick to find locations adjacent to ice tiles.
    ice_diff = np.diff(obs["board"]["ice"])
    pot_ice_spots = np.argwhere(ice_diff == 1)
    if len(pot_ice_spots) == 0:
        pot_ice_spots = potential_spawns

    # pick a random ice spot and search around it for spawnable locations.
    trials = 5
    while trials > 0:
        pos_idx = np.random.randint(0, len(pot_ice_spots))
        pos = pot_ice_spots[pos_idx]
        area = 3
        for x in range(area):
            for y in range(area):
                check_pos = [pos[0] + x - area // 2, pos[1] + y - area // 2]
                if tuple(check_pos) in potential_spawns_set:
                    done_search = True
                    pos = check_pos
                    break
            if done_search:
                break
        if done_search:
            break
        trials -= 1

    if not done_search:
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        pos = spawn_loc

    # this will spawn a factory at pos and with all the starting metal and water
    metal = obs["teams"][player]["metal"]
    return dict(spawn=pos, metal=metal, water=metal)


def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False
