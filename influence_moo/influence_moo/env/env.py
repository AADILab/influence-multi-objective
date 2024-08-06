from copy import deepcopy
import numpy as np
from influence_moo.evo.network import NN
from influence_moo.utils import out_of_bounds, determine_collision, line_of_sight

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

        self.path.append(deepcopy(self.position))
        self.h_path.append(deepcopy(self.h_position))
        self.surface_path.append(self.surfaced)

class ASV():
    def __init__(self, position, auvs, connectivity_grid, policy_network):
        self.position = position
        self.auvs = auvs
        self.connectivity_grid = connectivity_grid
        self.policy_network = policy_network
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
        return self.policy_network.forward(observation)

    def ping(self):
        for auv in self.auvs:
            if line_of_sight(self.position, auv.position, self.connectivity_grid, 0.1):
                auv.h_position = deepcopy(auv.position)


class OceanEnv():
    def __init__(self, mission, dt, num_asvs):
        self.mission = mission
        self.dt = dt
        self.num_asvs = num_asvs

    def run(self, num_iterations):
        # Let's give it a try
        paths = [self.mission.pathA, self.mission.pathB, self.mission.pathC, self.mission.pathD]
        self.auvs = [AUV(path, 1.) for path in paths]

        policy_network = NN(num_inputs=2+2*len(self.auvs)+self.mission.connectivity_grid.size, num_hidden=[20,20], num_outputs = 2)
        self.asv = ASV(position=self.mission.root_node[0].astype(float), auvs=self.auvs, connectivity_grid=self.mission.connectivity_grid, policy_network=policy_network)

        for _ in range(num_iterations):
            # Ping auvs
            self.asv.ping()

            # Move asv first
            if not self.asv.crashed:
                self.asv_velocity = self.asv.policy(self.asv.get_observation())
                new_position = self.asv.position + self.asv_velocity*self.dt
                if not out_of_bounds(new_position, self.mission.connectivity_grid.shape[0], self.mission.connectivity_grid.shape[1]):
                    self.asv.position += self.asv_velocity*self.dt

            if determine_collision(self.asv.position, self.mission.connectivity_grid):
                self.asv.crashed = True

            self.asv.path.append(deepcopy(self.asv.position))

            # Move auvs
            for id, auv in enumerate(self.auvs):
                # Ping all the auvs from the starting point
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
