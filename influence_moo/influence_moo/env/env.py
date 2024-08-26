import os
from copy import deepcopy
import numpy as np
from pathlib import Path
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
        self.action_history = []

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

        # Return the action the auv took
        return delta

class ASV():
    def __init__(self, position, auvs, connectivity_grid, policy_function):
        self.position = position
        self.auvs = auvs
        self.connectivity_grid = connectivity_grid
        self.policy_function = policy_function
        self.crashed = False
        self.path = [deepcopy(self.position)]
        self.action_history = []
        self.observation_history = []
        self.crash_history = []

    def policy(self, observation):
        # Output is vx, vy
        return self.policy_function(observation)

    def ping(self):
        for auv in self.auvs:
            if not auv.crashed and line_of_sight(self.position, auv.position, self.connectivity_grid, 0.1):
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

def remove_agent(agents, agent_ind):
    return agents[:agent_ind]+agents[agent_ind+1:]

class Rewards():
    """Rewards are based on AUV/ASV paths and influence heuristics"""
    def __init__(self, pois, connectivity_grid, collision_step_size, config
        ):
        """
        TODO: Counterfactual influence vs local influence
        influence_heuristic: line_of_sight, distance_threshold
        influence_type: all_or_nothing, granular
        auv_reward: global, local, difference
        asv_reward: global, local, difference, indirect_difference
        multi_reward: single, multiple
        """
        self.pois = pois
        self.connectivity_grid = connectivity_grid
        self.collision_step_size = collision_step_size

        self.influence_heuristic = config['rewards']['influence_heuristic']
        self.influence_type = config['rewards']['influence_type']
        self.auv_reward = config['rewards']['auv_reward']
        self.asv_reward = config['rewards']['asv_reward']
        self.multi_reward = config['rewards']['multi_reward']
        self.distance_threshold = config['rewards']['distance_threshold']

    def local_auv_reward(self, auv):
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

    def local_asv_reward(self, auvs, asv):
        # Maximize "local" influence on auvs
        # (Don't pay attention to what other asvs are doing)
        # (or even really what the AUVs are doing)
        reward = 0.
        for auv in auvs:
            for auv_position, asv_position in zip(auv.path, asv.path):
                reward += self.influence(asv_position, auv_position)
        return reward

    def get_nearest_auvs(self, auvs):
        # Initialize storage for which auv was closest to each poi
        nearest_auvs = [AUVInfo(auv_ind=None, distance=np.inf, position=None) for _ in self.pois]

        # Go through paths and determine which auv was closest to each poi
        for auv_ind, auv in enumerate(auvs):
            for auv_position in auv.path:
                for poi_ind, poi in enumerate(self.pois):
                    # Make sure auv was not counterfactually removed
                    if not np.isnan(auv_position[0]) and not np.isnan(auv_position[1]):
                        # Check line of sight
                        if line_of_sight(auv_position, poi.position, self.connectivity_grid, self.collision_step_size):
                            distance = np.linalg.norm(auv_position - poi.position)
                            if distance < nearest_auvs[poi_ind].distance:
                                nearest_auvs[poi_ind] = AUVInfo(auv_ind=auv_ind, distance = distance, position = deepcopy(auv_position))

        return nearest_auvs

    def global_(self, auvs, asvs):
        """Global reward for entire team"""
        # Reward is zero if anyone crashed
        # if any([auv.crashed for auv in auvs]+[asv.crashed for asv in asvs]):
        #     return 0.0

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
        elif self.influence_heuristic == "distance_threshold":
            if np.linalg.norm(auv_position-asv_position) < self.distance_threshold:
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
                    influence_array[i, a] += self.influence(asv.path[i], auv.path[i])

        return influence_array

    def counterfactual_influence(self, auvs, asvs, asv_ind):
        """Compute an influence array if we remove an ASV"""
        asvs_with_ind_removed = asvs[:asv_ind]+asvs[asv_ind+1:]
        return self.influence_array(auvs=auvs, asvs=asvs_with_ind_removed)

    def remove_influence(self, auvs, influence_array):
        """Remove auv states that are influenced by an asv according to influence array"""
        # MAKE A DEEPCOPY BECAUSE THE AUV OBJECTS WILL BE MODIFIED
        counterfactual_auvs = deepcopy(auvs)
        num_steps = influence_array.shape[0]
        num_auvs = influence_array.shape[1]
        for i in range(num_steps):
            for a in range(num_auvs):
                if influence_array[i, a] > 0.0:
                    counterfactual_auvs[a].path[i] = np.array([np.nan, np.nan])
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

    def remove_asv_influence(self, auvs, asv, asv_ind):
        pass

    def indirect_difference_auv(self, auvs, asvs, asv_ind, auv_ind, difference_reward):
        """Compute indirect difference reward for ASV on one AUV"""
        pass

    def compute(self, auvs, asvs):
        """
        influence_heuristic: line_of_sight, distance_threshold
        influence_type: all_or_nothing, granular
        auv_reward: global, local, difference
        asv_reward: global, local, difference, indirect_difference_team, indirect_difference_auv
        multi_reward: single, multiple
        """
        # Compute global reward
        G = self.global_(auvs=auvs, asvs=asvs)

        # Global reward for ASVs
        if self.asv_reward == "global":
            return [G for asv in asvs], G
        # Local reward for ASVs
        elif self.asv_reward == "local":
            return [self.local_asv_reward(auvs=auvs, asv=asv) for asv in asvs], G
        # Difference reward for ASVs
        elif self.asv_reward == "difference":
            # Removing an ASV's path does not actually change G
            # Hence, the difference reward is always 0.0
            return [0.0 for asv in asvs], G
        # Indirect difference reward for ASVs based on indirect contribution to team
        elif self.asv_reward == "indirect_difference_team" or self.asv_reward == "indirect_difference_auv":
            # Create influence array that tells us how much each AUV was influenced
            influence_array = self.influence_array(auvs=auvs, asvs=asvs)
            # Copy asvs with asv j removed
            asvs_minus_j_list = [
                remove_agent(asvs, asv_ind) for asv_ind in range(len(asvs))
            ]
            # Create counterfactual influence arrays when we remove each asv
            counterfactual_influence_list = [
                self.influence_array(auvs, asvs_minus_j) for asvs_minus_j in asvs_minus_j_list
            ]
            # Each asv gets an influence array that is (influence) - (counterfactual influence with that asv removed)
            influence_j_list = [
                influence_array - counterfactual_influence for counterfactual_influence in counterfactual_influence_list
            ]
            # Create counterfactual AUV paths with the influence of asv j removed
            auvs_minus_j_list = [
                self.remove_influence(auvs, influence_j) for influence_j in influence_j_list
            ]
            # Compute counterfactual G with the influence of asv j removed
            counterfactual_G_j_list = [
                self.global_(auvs_minus_j, asvs_minus_j) for auvs_minus_j, asvs_minus_j in zip(auvs_minus_j_list, asvs_minus_j_list)
            ]
            if self.asv_reward == "indirect_difference_team":
                # Finally compute an indirect difference reward with these counterfactual paths
                return [
                    G-counterfactual_G for counterfactual_G in counterfactual_G_j_list
                ], G

        # Rewards for AUVs that will be used to derive ASV rewards
        if self.auv_reward == "global":
            auv_rewards = [G for auv in auvs]
        elif self.auv_reward == "local":
            auv_rewards = [self.local_auv_reward(auv) for auv in auvs]
        elif self.auv_reward == "difference":
            # Copy paths with auv i removed
            auvs_minus_i_list = [remove_agent(auvs, auv_ind) for auv_ind in range(len(auvs))]
            # Counterfactual G for each removed auv
            counterfactual_G_remove_i_list = [
                self.global_(auvs=auvs_minus_i, asvs=asvs) for auvs_minus_i in auvs_minus_i_list
            ]
            # Compute D for each auv
            auv_rewards = [
                G-counterfactual_G for counterfactual_G in counterfactual_G_remove_i_list
            ]

        # Rewards for ASVs that are derived from AUV rewards
        if self.asv_reward == "indirect_difference_auv":
            if self.auv_reward == "global" or self.auv_reward == "local":
                pass
            elif self.auv_reward == "difference":
                # Decompose each individual auv reward into many rewards. One for each asv.
                decomposed_auv_rewards = []
                for auv_ind in range(len(auvs)):
                    auvs_minus_ij_list = [
                        remove_agent(auvs_minus_j, auv_ind) for auvs_minus_j in auvs_minus_j_list
                    ]
                    counterfactual_G_ij_list = [
                        self.global_(auvs_minus_ij, asvs_minus_j) for auvs_minus_ij, asvs_minus_j in zip(auvs_minus_ij_list, asvs_minus_j_list)
                    ]
                    difference_ij_list = [
                        G_j - G_ij for G_j, G_ij in zip(counterfactual_G_j_list, counterfactual_G_ij_list)
                    ]
                    indirect_difference_ij_list = [
                        auv_rewards[auv_ind] - D_ij for D_ij in difference_ij_list
                    ]
                    decomposed_auv_rewards.append(indirect_difference_ij_list)
                # Now map this to auv rewards
                asv_rewards = [[None for auv in auvs] for asv in asvs]
                for j in range(len(auvs)):
                    for i in range(len(asvs)):
                        asv_rewards[i][j] = decomposed_auv_rewards[j][i]
                if self.multi_reward == "multiple":
                    return asv_rewards, G
                elif self.multi_reward == "single":
                    return [sum(rewards_per_asv) for rewards_per_asv in asv_rewards], G

class OceanEnv():
    def __init__(self, config):
        mission_dir = Path(os.path.expanduser(Path(config["env"]["mission_dir"])))
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
        self.rewards = Rewards(self.pois, self.mission.connectivity_grid, self.collision_step_size, self.config)

    def get_asv_observation(self, asv_ind):
        # ASV has access to some but NOT ALL global state information
        """
        Note for development: Agent centric observations
        observations are like delta x's to other agents (more informative)
        cares about how far things are from itself (most likely)
        map information - raycast in 8 directions that probe how far a wall is (like a lidar)
        and that gives a lower dimensional representation but still a useful one
        """
        observation = [self.asvs[asv_ind].position[0], self.asvs[asv_ind].position[1]]
        other_asvs = remove_agent(self.asvs, asv_ind)
        for asv in other_asvs:
            observation.append(asv.position[0])
            observation.append(asv.position[1])
        for auv in self.auvs:
            # ASV has the same hypothesis about where AUVs are
            observation.append(auv.h_position[0])
            observation.append(auv.h_position[1])
        observation = np.array(observation)
        return observation

    def step(self):
        # Ping auvs
        for asv in self.asvs:
            if not asv.crashed:
                asv.ping()

        # Move asvs first
        for asv_ind, asv in enumerate(self.asvs):
            # Placeholder action
            asv_velocity = np.array([0.,0.])
            asv_observation = np.zeros(2*len(self.asvs)+2*len(self.auvs)+self.mission.connectivity_grid.size)
            if not asv.crashed:
                asv_observation = self.get_asv_observation(asv_ind)
                asv_velocity = asv.policy(asv_observation)
                asv.position += asv_velocity*self.dt

            if out_of_bounds(asv.position, self.mission.connectivity_grid.shape[0], self.mission.connectivity_grid.shape[1]) \
                or determine_collision(asv.position, self.mission.connectivity_grid):
                asv.crashed = True

            asv.path.append(deepcopy(asv.position))
            asv.action_history.append(asv_velocity)
            asv.observation_history.append(asv_observation)
            asv.crash_history.append(asv.crashed)

        # Move auvs
        for auv in self.auvs:
            # Placeholder action
            auv_action = np.array([0.,0.])
            # Wave moves auv
            if not auv.crashed:
                auv.position += np.array([ self.mission.wave_x(auv.position[0])*self.dt, self.mission.wave_y(auv.position[1])*self.dt ])
            if out_of_bounds(auv.position, self.mission.connectivity_grid.shape[0], self.mission.connectivity_grid.shape[1]) \
                or determine_collision(auv.position, self.mission.connectivity_grid):
                auv.crashed = True
            # auv acts based on hypothesis position
            if not auv.crashed:
                auv_action = auv.update(self.dt)
            if out_of_bounds(auv.position, self.mission.connectivity_grid.shape[0], self.mission.connectivity_grid.shape[1]) \
                or determine_collision(auv.position, self.mission.connectivity_grid):
                auv.crashed = True

            auv.path.append(deepcopy(auv.position))
            auv.h_path.append(deepcopy(auv.h_position))
            auv.action_history.append(auv_action)
            auv.surface_history.append(auv.surfaced)
            auv.crash_history.append(auv.crashed)

    def run(self, asv_policy_functions):
        # Let's give it a try
        self.auvs = [AUV(path, 1.) for path in self.mission.paths]

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
