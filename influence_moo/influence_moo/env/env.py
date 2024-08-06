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
        self.surface_path = [self.surfaced]

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
        self.surface_path.append(self.surfaced)

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
    def __init__(self, pois, connectivity_grid, collision_step_size):
        self.pois = pois
        self.connectivity_grid = connectivity_grid
        self.collision_step_size = collision_step_size

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
        # We need to go through the paths and determine which auv was closest to each poi
        nearest_auvs = self.get_nearest_auvs(auvs)

        # Now let's compute the value of the observations
        reward = 0
        for poi, auv_info in zip(self.pois, nearest_auvs):
            # Make sure this poi was observed
            if auv_info.auv_ind is not None and auv_info.distance < poi.observation_radius:
                reward += 1./np.max((auv_info.distance, 1.)) * poi.value
        return reward

    def difference(self, auvs, remove_ind, team_reward):
        auvs_with_ind_removed = auvs[:remove_ind]+auvs[remove_ind+1:]
        return team_reward - self.team(auvs_with_ind_removed)

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
