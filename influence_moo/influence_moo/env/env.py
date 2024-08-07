from copy import deepcopy
import numpy as np
from pathlib import Path
from influence_moo.evo.network import NN
from influence_moo.utils import out_of_bounds, determine_collision, line_of_sight
from influence_moo.env.mission import Mission

class AUV():
    def __init__(self, targets, max_velocity):
        self.max_velocity = max_velocity
        self.targets = deepcopy(targets)
        self.position = self.targets[0]
        # hypothesis position
        self.h_position = deepcopy(self.position)
        self.target_id = 1
        self.crashed = False
        self.surfaced = False
        self.path = [deepcopy(self.position)]
        self.h_path = [deepcopy(self.h_position)]
        self.surface_history = [self.surfaced]
        self.crash_history = [self.crashed]

    def update(self, dt):
        # Am I at my target?
        if np.allclose(self.h_position, self.targets[self.target_id]):
            # Only update target if this is not the last target
            if self.target_id < len(self.targets)-1:
                # New target
                self.target_id += 1
        # Delta to target
        delta = self.targets[self.target_id] - self.h_position
        # Don't exceed max velocity
        if np.linalg.norm(delta) > (self.max_velocity*dt):
            theta = np.arctan2(delta[1], delta[0])
            delta[0] = self.max_velocity*dt*np.cos(theta)
            delta[1] = self.max_velocity*dt*np.sin(theta)
        # Update position
        self.position += delta
        self.h_position += delta
        # Am I at my target?
        if np.allclose(self.h_position, self.targets[self.target_id]):
            # Only update target if this is not the last target
            if self.target_id < len(self.targets)-1:
                # New target
                self.target_id += 1
            # Otherwise, I have reached my last target, and it is time to surface
            else:
                self.surfaced = True
                # Now that I have surfaced,
                # I can use my GPS and no longer have uncertainty about my position
                self.h_position = self.position

        self.path.append(deepcopy(self.position))
        self.h_path.append(deepcopy(self.h_position))
        self.surface_history.append(self.surfaced)
        self.crash_history.append(self.crashed)

class ASV():
    def __init__(self, position, auvs, connectivity_grid, policy_function):
        self.position = position
        self.auvs = auvs
        self.connectivity_grid = connectivity_grid
        self.policy_function = policy_function
        self.crashed = False
        self.path = [deepcopy(self.position)]

    def get_observation(self):
        # ASV has the same hypothesis about where AUVs are
        observation = []
        observation.append(self.position[0])
        observation.append(self.position[1])
        for auv in self.auvs:
            observation.append(auv.position[0])
            observation.append(auv.position[1])
        observation = np.array(observation)
        observation = np.concatenate([observation, self.connectivity_grid.flatten()])
        return observation

    def policy(self, observation):
        # Output is vx, vy
        return self.policy_function(observation)

    def ping(self):
        for auv in self.auvs:
            if line_of_sight(self.position, auv.position, self.connectivity_grid, 0.1):
                auv.h_position = deepcopy(auv.position)

class POI():
    def __init__(self, position, value, observation_radius):
        self.position = position
        self.value = value
        self.observation_radius = observation_radius

class AUVInfo():
    def __init__(self, auv_ind, distance, position):
        self.auv_ind = auv_ind
        self.distance = distance
        self.position = position

class Rewards():
    """Rewards are based on AUV paths and influence heuristics"""
    def __init__(self, pois, connectivity_grid, collision_step_size, influence_heuristic):
        self.pois = pois
        self.connectivity_grid = connectivity_grid
        self.collision_step_size = collision_step_size
        self.influence_heuristic = influence_heuristic

    def local(self, auv):
        # No reward for crashing
        if auv.crashed:
            return 0.0

        min_distance = np.inf
        for auv_position in auv.path:
            for poi in self.pois:
                # Check line of sight
                if line_of_sight(auv_position, poi.position, self.connectivity_grid, self.collision_step_size):
                    distance = np.linalg.norm(auv_position - poi.position)
                    if distance < min_distance:
                        min_distance = distance

        return 1./np.max((distance, 1.)) * poi.value

    def get_nearest_auvs(self, auvs):
        # Initialize storage for which auv was closest to each poi
        nearest_auvs = [AUVInfo(auv_ind=None, distance=np.inf, position=None) for _ in self.pois]

        # Go through paths and determine which auv was closest to each poi
        for auv_ind, auv in enumerate(auvs):
            # This auv only counts if it didn't crash
            if not auv.crashed:
                for auv_position in auv.path:
                    for poi_ind, poi in enumerate(self.pois):
                        # Check line of sight
                        if line_of_sight(auv_position, poi.position, self.connectivity_grid, self.collision_step_size):
                            distance = np.linalg.norm(auv_position - poi.position)
                            if distance < nearest_auvs[poi_ind].distance:
                                nearest_auvs[poi_ind] = AUVInfo(auv_ind=auv_ind, distance = distance, position = deepcopy(auv_position))

        return nearest_auvs

    def team(self, auvs):
        """Team reward for entire team"""
        # We need to go through the paths and determine which auv was closest to each poi
        nearest_auvs = self.get_nearest_auvs(auvs)

        # Now let's compute the value of the observations
        reward = 0
        for poi, auv_info in zip(self.pois, nearest_auvs):
            # Make sure this poi was observed
            if auv_info.auv_ind is not None and auv_info.distance < poi.observation_radius:
                reward += 1./np.max((auv_info.distance, 1.)) * poi.value
        return reward

    def difference(self, auvs, auv_ind, team_reward):
        """Difference reward for a single AUV"""
        auvs_with_ind_removed = auvs[:auv_ind]+auvs[auv_ind+1:]
        return team_reward - self.team(auvs_with_ind_removed)

    def influence(self, asv_position, auv_position):
        """Compute an influence heuristic telling us the influence of an ASV on an AUV"""
        # Binary influence computation. If you have line of sight, yes influence. No line of sight, no influence
        if self.influence_heuristic == "line_of_sight":
            if line_of_sight(auv_position, asv_position, self.connectivity_grid, self.collision_step_size):
                return 1.0
            else:
                return 0.0

    def influence_array(self, auvs, asvs):
        """Compute array that tells us how much support each auv received at each step"""
        # Init array
        num_steps = len(auvs[0].path)
        num_auvs = len(auvs)
        influence_array = np.zeros((num_steps, num_auvs))

        # Iterate through trajectories
        for i in range(num_steps):
            # Tell me how much each auv was supported at this step
            for a, auv in enumerate(auvs):
                for asv in asvs:
                    influence_array[i, a] += self.influence(asv.position, auv.position)

        return influence_array

    def counterfactual_influence(self, auvs, asvs, asv_ind):
        """Compute an influence array if we remove an ASV"""
        asvs_with_ind_removed = asvs[:asv_ind]+asvs[asv_ind+1:]
        return self.influence_array(auvs=auvs, asvs=asvs_with_ind_removed)

    def counterfactual_influenced_agents(self, auvs, influence_array):
        """Remove auv states that are influenced by an asv according to influence array"""
        counterfactual_auvs = deepcopy(auvs)
        num_steps = influence_array.shape[0]
        num_auvs = influence_array.shape[1]
        for i in range(num_steps):
            for a in range(num_auvs):
                if influence_array[i, a] > 0.0:
                    counterfactual_auvs[a].path[i] = np.array([np.inf, np.inf])
        return counterfactual_auvs

    def influence_difference(self, auvs, asvs, asv_ind):
        return self.influence_array(auvs=auvs, asvs=asvs) - \
            self.counterfactual_influence(auvs=auvs, asvs=asvs, asv_ind=asv_ind)

    def indirect_difference_team(self, auvs, asvs, asv_ind, team_reward):
        """Compute indirect difference reward for ASV on team reward"""
        # Influence array for just this ASV
        influence_array_asv = self.influence_difference(auvs=auvs,asvs=asvs, asv_ind=asv_ind)

        # Generate counterfactual auvs that have states removed where they are influenced
        counterfactual_auvs = self.counterfactual_influenced_agents(auvs, influence_array_asv)
        return team_reward - self.team(counterfactual_auvs)


    # def compute_influence_history(self, asv, auv):
    #     """Compute influence heuristic telling us when ASV is supporting AUV"""
    #     influence_history = []
    #     for auv_position, asv_position in zip(auv.path, asv.path):
    #         influence_history.append(self.compute_influence(asv_position, auv_position))
    #     return influence_history

    # def compute_influence_history

    # def indirect_difference(self, asvs, auvs, asv_ind, difference_rewards):
    #     """Indirect difference reward vector for single ASV across multiple AUVs"""
    #     counterfactual_auvs = deepcopy(auvs)

    #     # Now remove


class OceanEnv():
    def __init__(self, config):
        mission_dir = Path(config["root_dir"]) / "missions" / config["env"]["mission"]
        self.mission = Mission(mission_dir)

        self.config = config
        ec = self.config["env"]
        self.dt = ec["dt"]
        self.t_final = ec["t_final"]
        self.num_iterations = int(self.t_final / self.dt)
        self.asv_max_speed = ec['asv']['max_speed']
        self.auv_max_speed = ec['asv']['max_speed']
        self.collision_step_size = ec['collision_step_size']
        self.pois = []
        for poi_position, poi_config in zip(self.mission.pois, ec['pois']):
            self.pois.append(POI(position=poi_position, value=poi_config['value'], observation_radius=poi_config['observation_radius']))
        self.rewards = Rewards(self.pois, self.mission.connectivity_grid, self.collision_step_size)

    def step(self):
        # Ping auvs
        for asv in self.asvs:
            asv.ping()

        # Move asvs first
        for asv in self.asvs:
            if not asv.crashed:
                asv_velocity = asv.policy(asv.get_observation())
                new_position = asv.position + asv_velocity*self.dt
                if not out_of_bounds(new_position, self.mission.connectivity_grid.shape[0], self.mission.connectivity_grid.shape[1]):
                    asv.position += asv_velocity*self.dt

            if determine_collision(asv.position, self.mission.connectivity_grid):
                asv.crashed = True

            asv.path.append(deepcopy(asv.position))

        # Move auvs
        for auv in self.auvs:
            # Wave moves auv
            if not auv.crashed:
                auv.position += np.array([ self.mission.wave_x(auv.position[0])*self.dt, self.mission.wave_y(auv.position[1])*self.dt ])
            if determine_collision(auv.position, self.mission.connectivity_grid):
                auv.crashed = True
            # auv acts based on hypothesis position
            if not auv.crashed:
                auv.update(self.dt)
            if determine_collision(auv.position, self.mission.connectivity_grid):
                auv.crashed = True

    def run(self, asv_policy_functions):
        # Let's give it a try
        paths = [self.mission.pathA, self.mission.pathB, self.mission.pathC, self.mission.pathD]
        self.auvs = [AUV(path, 1.) for path in paths]

        self.asvs = [
            ASV(
                position=self.mission.root_node[0].astype(float),
                auvs=self.auvs,
                connectivity_grid=self.mission.connectivity_grid,
                policy_function=policy_func
            )
            for policy_func in asv_policy_functions
        ]

        for _ in range(self.num_iterations):
            self.step()
