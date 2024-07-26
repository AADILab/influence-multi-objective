from copy import deepcopy, copy
import numpy as np

class Env():
    def __init__(self, agents, pois, dt, max_velocity):
        self.agents = agents
        self.pois = pois
        self.dt = dt
        self.max_velocity = max_velocity

    def update(self):
        pass

class Agent():
    def __init__(self, position):
        self.position = position

    def policy(self, observation):
        pass

    def observe(self, env):
        pass

# Surface vehicle tells each underwater vehicle what to do
# and provides support to the vehicles by staying near them
# Let's say each surface vehicle has a certain number of
# underwater vehicles to coordinate that start at
# surface vehicle and come back
class SurfaceVehicle(Agent):
    def __init__(self, position):
        super.__init__(self, position)

    def policy(self, observation):
        pass

    def observe(self, env):
        pass

# Underwater vehicle goes to POI
# And takes time to observe it.
# Time depends on how close surface vehicle is
class UnderwaterVehicle(Agent):
    def __init__(self, position, poi_list, fail_probability, loss_probability):
        super.__init__(self, position)
        # List of POIs you need to observe
        self.poi_list = poi_list
        # What POI are you working on right now
        self.current_poi_id = self.poi_list[0]
        # Whether to return home from a mission complete
        self.done = False
        # probability the robot will fail and need help
        self.fail_probability = fail_probability
        # whether robot needs help
        self.sos = False
        # probability the robot is lost once it fails
        self.loss_probability = loss_probability

    def nav(self, desired_position, env):
        # There will be obstacles so you can't go straight where you want
        # you have to do some navigation

    def policy(self, env):
        # Compute an action for the vehicle
        # Check if the POI you are working on is complete
        if env.pois[self.current_poi_id].status() == 1.0:
            # Move onto the next POI
            self.current_poi_id += 1
            # Or go to nearest surface vehicle for retreival
            if self.current_poi_id >= len(self.poi_list):
                self.done = True

        # Compute where you want to be
        if self.done:
            # Find nearest surface vehicle and go there
            nearest_surface_vehicle = self.nearestSurfaceVehicle(env.agents)
            desired_position = nearest_surface_vehicle.position
        # Or go to your current poi
        else:
            S

        # Go to your poi
        desired_delta = env.pois[self.current_poi_id] - self.position
        # Bound action by
        pass

    def observe(self, env):
        pass

    def nearestSurfaceVehicle(self, agents):
        nearest_support = None
        closest_distance = None
        for agent in agents:
            if type(agent) == SurfaceVehicle:
                if nearest_support is None:
                    closest_distance = distance(self.position, agent.position)
                    nearest_support = agent
                else:
                    dist = distance(self.position, agent.position)
                    if dist < closest_distance:
                        closest_distance = dist
                        nearest_support = agent
        return nearest_support, closest_distance

    def computeSupport(self, agents):
        _, closest_distance = self.nearestSurfaceVehicle(agents)
        # Support is inversely proportional to distance
        # The closer you are, the better the support
        if closest_distance > 0.0:
            support = 1./closest_distance
            # But support cannot be infinite
            if support > 1.0:
                support = 1.0
        # Divide by zero gives infinity. Avoid this
        else:
            support = 1.0
        return support

class POI():
    def __init__(self, position, observation_radius, observation_rate, value, id):
        self.position = position
        self.status = 0.0
        # How close does vehicle have to be to work on this task
        self.observation_radius = observation_radius
        # How difficult is this POI to observe?
        # Difficult POIs take longer
        self.observation_rate = observation_rate
        self.id = id
        self.value = value

    def update(self, env):
        # Check if agent is working on POI
        for agent in env.agents:
            if type(agent) == UnderwaterVehicle \
                and self.id == agent.current_poi_id \
                    and distance(self.position, agent.position) <= self.observation_radius:
                # Compute an update to this POI
                self.status += self.difficulty*env.dt*agent.computeSupport(env.agents)
                # Can't be more done than done
                if self.status > 1.0:
                    self.status = 1.0
        # Just updating internal state
        return None

    # How far is this POI from being observed. 0-1
    # 0, not started
    # 1, complete
    def status(self):
        return copy(self.status)

def distance(pos0, pos1):
    return np.linalg.norm(pos0-pos1)
