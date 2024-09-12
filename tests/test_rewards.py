"""
Unit tests to check reward functions work how I expect them to

NOTE: TESTS ARE NOT EXTENSIVE!!! Bugs can sneak past them.
No guarantees of bug-free code just because the unit tests are happy

To run individual tests:
python tests/test_rewards.py TestRewards.test_spoof_0b
or
python -m unittest tests.test_rewards.TestRewards.test_spoof_0b

TODO: Include checks for G_vec in the tests. Currently the tests do not check G_vec is correct
"""
import unittest
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from influence_moo.env.env import Rewards, AUV, ASV, POI, remove_agent
from influence_moo.plotting import plot_grid, plot_pts
from influence_moo.utils import check_path

class Spoof():
    def __init__(self, pois, auvs, asvs, connectivity_grid, collision_step_size):
        self.pois = pois
        self.auvs = auvs
        self.asvs = asvs
        self.connectivity_grid = connectivity_grid
        self.collision_step_size = collision_step_size

class TestRewards(unittest.TestCase):
    def setUp(self):
        self.VISUALIZE = False

    def setup_spoof_0(self):
        # Spoof a rollout
        connectivity_grid = np.ones((20, 10))
        connectivity_grid[10,:] = 0.0
        connectivity_grid[5,2:8] = 0.0
        connectivity_grid[10:,3] = 0.0

        pois = [
            POI(position=np.array([1,9]), value=0.3, observation_radius=2.0),
            POI(position=np.array([1,1]), value=2.5, observation_radius=1.0),
            POI(position=np.array([16,8]), value=1.2, observation_radius = 1.0),
            POI(position=np.array([19,2]), value=3.7, observation_radius = 1.0)
        ]
        poi_positions = np.array([poi.position for poi in pois])

        auvs = [
            AUV(targets=[None], max_velocity=None),
            AUV(targets=[None], max_velocity=None),
            AUV(targets=[None], max_velocity=None)
        ]

        auvs[0].path = np.array([
            [1,1],
            [1,2],
            [1,5],
            [1,8],
            [1,9]
        ], dtype=np.float32)

        auvs[1].path = np.array([
            [12,1],
            [12,2],
            [14,2],
            [16,2],
            [19,2]
        ], dtype=np.float32)

        auvs[2].path = np.array([
            [12,5],
            [14,5],
            [16,5],
            [16,6],
            [16,8]
        ], dtype=np.float32)

        for id, auv in enumerate(auvs):
            # Track each auv during testing
            auv.id = id
            # Add crash history
            for _ in auv.path[:-1]:
                auv.crash_history.append(False)

        asvs = [
            ASV(position=None, auvs=auvs, connectivity_grid=None, policy_function=None),
            ASV(position=None, auvs=auvs, connectivity_grid=None, policy_function=None)
        ]

        for id, asv in enumerate(asvs):
            # Track asvs during testing
            asv.id = id

        asvs[0].path = np.array([[8,8] for _ in range(5)], dtype=np.float32)
        asvs[1].path = np.array([[13,8] for _ in range(5)], dtype=np.float32)

        if self.VISUALIZE:
            fig, ax = plt.subplots(1,1,dpi=100)
            plot_grid(connectivity_grid, cmap='tab10_r')
            plot_pts(poi_positions, ax, marker='o', fillstyle='none', linestyle='none',color='tab:green')
            plot_pts(auvs[0].path, ax, ls=(0, (1,2)), color='pink', lw=1)
            plot_pts(auvs[1].path, ax, ls='dashed', color='purple', lw=1)
            plot_pts(auvs[2].path, ax, ls='dashdot', color='tab:cyan', lw=1)
            plot_pts(asvs[0].path, ax, marker='+', color='orange')
            plot_pts(asvs[1].path, ax, marker='+', color='tab:cyan')
            plot_pts(np.array([auvs[0].path[-2]]), ax, marker='x', color='pink')
            ax.set_title("Rollout for Testing Rewards")
            plt.show()

        self.spoof_0 = Spoof(pois, auvs, asvs, connectivity_grid, collision_step_size=0.1)

    def setup_spoof_1(self):
        """This rollout tests if an AUV crashes"""
        connectivity_grid = np.ones((20, 10))
        connectivity_grid[10,:] = 0.0
        connectivity_grid[5,2:8] = 0.0
        connectivity_grid[10:,3] = 0.0

        pois = [
            POI(position=np.array([1,9]), value=0.3, observation_radius=2.0),
            POI(position=np.array([1,1]), value=2.5, observation_radius=1.0),
            POI(position=np.array([16,8]), value=1.2, observation_radius = 1.0),
            POI(position=np.array([19,2]), value=3.7, observation_radius = 1.0)
        ]
        poi_positions = np.array([poi.position for poi in pois])

        auvs = [
            AUV(targets=[None], max_velocity=None),
            AUV(targets=[None], max_velocity=None),
            AUV(targets=[None], max_velocity=None)
        ]

        # For tracking auvs during testing
        for id, auv in enumerate(auvs):
            auv.id = id

        auv0_xs = np.linspace(1.,1.,100)
        auv0_ys = np.linspace(1.,9.,100)
        auvs[0].path = np.array([auv0_xs, auv0_ys]).T

        auv1_xs = np.linspace(12.,19.,100)
        auv1_ys = np.linspace(1.,2.,100)
        auvs[1].path = np.array([auv1_xs, auv1_ys]).T

        auv2_xs = np.linspace(12.,16.,100)
        auv2_ys = np.linspace(5.,8.,100)
        auvs[2].path = np.array([auv2_xs, auv2_ys]).T

        asvs = [
            ASV(position=None, auvs=auvs, connectivity_grid=None, policy_function=None),
            ASV(position=None, auvs=auvs, connectivity_grid=None, policy_function=None)
        ]

        # For tracking these asvs during testing
        for id, asv in enumerate(asvs):
            asv.id = id

        asv0_xs = np.linspace(8.,8.,100)
        asv0_ys = np.linspace(8.,8.,100)
        asvs[0].path = np.array([asv0_xs, asv0_ys]).T

        asv1_xs = np.linspace(13,13,100)
        asv1_ys = np.linspace(8,8,100)
        asvs[1].path = np.array([asv1_xs, asv1_ys]).T

        if self.VISUALIZE:
            fig, ax = plt.subplots(1,1,dpi=100)
            plot_grid(connectivity_grid, cmap='tab10_r')
            plot_pts(poi_positions, ax, marker='o', fillstyle='none', linestyle='none',color='tab:green')
            plot_pts(auvs[0].path, ax, ls=(0, (1,2)), color='pink', lw=1)
            plot_pts(auvs[1].path, ax, ls='dashed', color='purple', lw=1)
            plot_pts(auvs[2].path, ax, ls='dashdot', color='tab:cyan', lw=1)
            plot_pts(asvs[0].path, ax, marker='+', color='orange')
            plot_pts(asvs[1].path, ax, marker='+', color='tab:cyan')
            plot_pts(np.array([auvs[0].path[-2]]), ax, marker='x', color='pink')
            ax.set_title("Rollout for Testing Rewards")
            plt.show()

        self.spoof_0 = Spoof(pois, auvs, asvs, connectivity_grid, collision_step_size=0.1)

    def get_spoof_0(self):
        if not "spoof_0" in self.__dict__:
            self.setup_spoof_0()
        return self.spoof_0

    def test_spoof_0a(self):
        """Test granular indirect difference reward to auvs' difference rewards as multiple rewards

        I'm going to breakdown the rewards process and see if the output is correct at each step
        """
        spoof_0 = self.get_spoof_0()
        pois, auvs, asvs, connectivity_grid, collision_step_size = \
            spoof_0.pois, spoof_0.auvs, spoof_0.asvs, spoof_0.connectivity_grid, spoof_0.collision_step_size
        config = {
            "rewards":
            {
                "influence_heuristic": "line_of_sight",
                "influence_type": "granular",
                "trajectory_influence_threshold": 0.0,
                "auv_reward": "difference",
                "asv_reward": "indirect_difference_auv",
                "multi_reward": "multiple",
                "distance_threshold": 0.0
            }
        }

        rewards = Rewards(
            pois = pois,
            connectivity_grid = connectivity_grid,
            collision_step_size = collision_step_size,
            config = config
        )

        # Global reward
        G, _ = rewards.global_(auvs=auvs, asvs=asvs)
        total_poi_value = sum([poi.value for poi in pois])+pois[0].value+pois[1].value
        self.assertTrue(G == total_poi_value, "Global reward computed incorrectly")

        # Influence
        influence_array = rewards.influence_array(auvs=auvs, asvs=asvs)
        iv = np.zeros( (len(auvs[0].path), len(auvs)) )
        # auv 0 was influenced in last 2 timesteps
        iv[-2:, 0] = 1.
        # auv 2 was influenced in entire trajectory
        iv[:,2] = 1.
        self.assertTrue(np.allclose(influence_array , iv), "Influence array computed incorrectly")

        # Remove asvs
        asvs_minus_j_list = [
            remove_agent(asvs, asv_ind) for asv_ind in range(len(asvs))
        ]
        ids = []
        for asvs_minus_j in asvs_minus_j_list:
            ids.append(asvs_minus_j[0].id)
        correct_removal = False
        if len(ids) == 2 and ids[0] == 1 and ids[1] == 0:
            correct_removal = True
        self.assertTrue(correct_removal, "ASVs removed incorrectly")

        # Create counterfactuals where we remove influence
        counterfactual_influence_list = [
            rewards.influence_array(auvs, asvs_minus_j) for asvs_minus_j in asvs_minus_j_list
        ]
        cil = [np.zeros(influence_array.shape), np.zeros(influence_array.shape)]
        cil[0][:,2] = 1.
        cil[1][-2:,0] = 1.
        correct = True
        for c, ci in zip(counterfactual_influence_list, cil):
            if not np.allclose(c,ci):
                correct = False
        self.assertTrue(correct, "Counterfactual influence array computed incorrectly")

        # Compute the influence of each asv
        influence_j_list = [
            influence_array - counterfactual_influence for counterfactual_influence in counterfactual_influence_list
        ]
        # Influence of asv 0
        i0 = np.zeros(influence_array.shape)
        i0[-2:,0] = 1.
        # Influence of asv 1
        i1 = np.zeros(influence_array.shape)
        i1[:,2] = 1.
        # Check
        correct = True
        for inf, i in zip(influence_j_list, [i0,i1]):
            if not np.allclose(inf,i):
                correct = False
        self.assertTrue(correct, "Influence of ASVs computed incorrectly")

        # Now we remove the influence of each asv
        auvs_minus_j_list = [
            rewards.remove_influence(auvs, influence_j) for influence_j in influence_j_list
        ]

        # When we remove asv 0, we remove the last observation of auv 0
        auvs_minus_0 = deepcopy(auvs)
        auvs_minus_0[0].path[-2:,:] = np.nan

        # When we remove asv 1, we remove the entire path of auv 2
        auvs_minus_1 = deepcopy(auvs)
        auvs_minus_1[2].path[:,:] = np.nan

        auvs_minus_j_test = [auvs_minus_0, auvs_minus_1]
        correct = True
        for auvs_act, auvs_test in zip(auvs_minus_j_list, auvs_minus_j_test):
            for auv_act, auv_test in zip(auvs_act, auvs_test):
                if not check_path(auv_act.path, auv_test.path):
                    correct=False
                    break
        self.assertTrue(correct, "ASV influences removed incorrectly")

        # Compute counterfactual G with asv influence removed
        counterfactual_G_j_list = [
            rewards.global_(auvs=auvs_minus_j,asvs=asvs)[0] for auvs_minus_j in auvs_minus_j_list
        ]
        test_list = [
            rewards.global_(auvs=auvs_minus_j, asvs=asvs)[0] for auvs_minus_j in auvs_minus_j_test
        ]
        target = [9.9, 9.3]
        # Make sure constructed counterfactual list matches our test generated list and the target list
        correct = np.allclose(counterfactual_G_j_list, test_list) and np.allclose(counterfactual_G_j_list, target)
        self.assertTrue(correct, "Counterfactual G with asv influences removed computed incorrectly")

        # Compute D-Indirect according to team contribution
        indirect_difference_reward_team = [
            G-counterfactual_G for counterfactual_G in counterfactual_G_j_list
        ]
        target = [0.6, 1.2]
        self.assertTrue(np.allclose(indirect_difference_reward_team, target), "Indirect Difference Reward based on contribution to team computed incorrectly")

        """Difference rewards for AUVs"""
        # Remove auv i
        auvs_minus_i_list = [remove_agent(auvs, auv_ind) for auv_ind in range(len(auvs))]
        test_list = [[1,2], [0,2], [0,1]]
        correct = True
        for auvs_minus_i, test_ids in zip(auvs_minus_i_list, test_list):
            auv_ids = [auv.id for auv in auvs_minus_i]
            if not np.allclose(auv_ids, test_ids):
                correct = False
                break
        self.assertTrue(correct, "AUVs removed incorrectly for AUV Difference Rewards")

        # Counterfactual G for each removed auv i
        counterfactual_G_remove_i_list = [
            rewards.global_(auvs=auvs_minus_i, asvs=asvs)[0] for auvs_minus_i in auvs_minus_i_list
        ]
        target = [4.9, 6.8, 9.3]
        self.assertTrue(np.allclose(counterfactual_G_remove_i_list, target), "Counterfactual G for removed AUVs computed incorrectly")

        # D for each auv i
        auv_rewards = [
            G-counterfactual_G for counterfactual_G in counterfactual_G_remove_i_list
        ]
        target = [5.6, 3.7, 1.2]
        self.assertTrue(np.allclose(auv_rewards, target), "Difference Rewards for AUVs computed incorrectly")

        """ Decompose each individual auv reward into many rewards. One for each asv. """
        # Remove auv 0 from each set we removed asv j's influence from
        auv_ind = 0
        auvs_minus_ij_list = [
            remove_agent(auvs_minus_j, auv_ind) for auvs_minus_j in auvs_minus_j_list
        ]

        # Create counterfactuals manually to check if they match the automatically computed ones

        # When we remove influence of asv0, auv0 is the only one affected. So this is the same as
        # just removing auv0
        remove_asv0_auv0_auvpaths = [deepcopy(auv.path) for auv in auvs]
        remove_asv0_auv0_auvpaths = remove_asv0_auv0_auvpaths[1:]

        # When we remove influence of asv1, auv2 is affected. Change auv2's path to nans
        # Then remove auv0 entirely
        remove_asv1_auv0_auvpaths = [deepcopy(auv.path) for auv in auvs]
        remove_asv1_auv0_auvpaths[2][:,:] = np.nan
        remove_asv1_auv0_auvpaths = remove_asv1_auv0_auvpaths[1:]

        # Check if these counterfactual paths are correct
        correct_paths_asv0_auv0 = True
        for auv, test_path in zip(auvs_minus_ij_list[0], remove_asv0_auv0_auvpaths):
            if not check_path(auv.path, test_path):
                correct_paths_asv0_auv0 = False
                break

        correct_paths_asv1_auv0 = True
        for auv, test_path in zip(auvs_minus_ij_list[1], remove_asv1_auv0_auvpaths):
            if not check_path(auv.path, test_path):
                correct_paths_asv1_auv0 = False
                break

        self.assertTrue(correct_paths_asv0_auv0 and correct_paths_asv1_auv0, \
            "Counterfactual paths removing asv influences and auv0 computed incorrectly")

        # Continue with computing counterfactual G with the removed asv j and auv 0
        counterfactual_G_ij_list = [
            rewards.global_(auvs=auvs_minus_ij,asvs=asvs)[0] for auvs_minus_ij in auvs_minus_ij_list
        ]
        target = [4.9, 3.7]
        self.assertTrue(np.allclose(counterfactual_G_ij_list, target), \
            "Counterfactual Gij for removing asv influences and auv0 computed incorrectly")

        # Compute difference of (removing asv j's influence) - (removing asv j's influence and auv 0)
        difference_ij_list = [
            G_j - G_ij for G_j, G_ij in zip(counterfactual_G_j_list, counterfactual_G_ij_list)
        ]
        target = [5.0, 5.6]
        self.assertTrue(np.allclose(difference_ij_list, target), \
            "Difference between removing (asv j) and (asv j and auv0) computed incorrectly")

        # Compute indirect difference reward (D of auv0) - (D of auv0 and each asv)
        indirect_difference_ij_list = [
            auv_rewards[auv_ind] - D_ij for D_ij in difference_ij_list
        ]
        target = [0.6, 0]
        self.assertTrue(np.allclose(indirect_difference_ij_list, target), \
            "Indirect Difference Reward for asvs relative to auv0 computed incorrectly")

        # Skip manually checking auv_ind 1 and 2
        # Check the reward computation against an expected reward
        expected_asv_rewards = [
            [0.6, 0, 0],
            [0, 0, 1.2]
        ]
        actual_asv_rewards, actual_G, _ = rewards.compute(auvs, asvs)
        self.assertTrue(np.allclose(actual_asv_rewards, expected_asv_rewards), \
            "Automatically computed Indirect Difference Rewards based on individual AUVs does not match expected rewards")
        self.assertTrue(G==actual_G, "G in rewards.compute() does not match G from rewards._global()")

    def test_spoof_0b(self):
        """Test granular indirect difference reward from auvs' difference rewards as single rewards"""
        spoof_0 = self.get_spoof_0()
        pois, auvs, asvs, connectivity_grid, collision_step_size = \
            spoof_0.pois, spoof_0.auvs, spoof_0.asvs, spoof_0.connectivity_grid, spoof_0.collision_step_size
        config = {
            "rewards":
            {
                "influence_heuristic": "line_of_sight",
                "influence_type": "granular",
                "trajectory_influence_threshold": 0.0,
                "auv_reward": "difference",
                "asv_reward": "indirect_difference_auv",
                "multi_reward": "single",
                "distance_threshold": 0.0
            }
        }

        rewards = Rewards(
            pois = pois,
            connectivity_grid = connectivity_grid,
            collision_step_size = collision_step_size,
            config = config
        )

        expected_out = [0.6, 1.2]
        actual_out, G, _ = rewards.compute(auvs, asvs)
        self.assertTrue(np.array(actual_out).shape == (2,),
            "Expected single reward for each asv. This is multiple rewards per asv")
        self.assertTrue(np.allclose(actual_out, expected_out),
            "Computed asv rewards do not match expected asv rewards")

    def test_spoof_0c(self):
        """Test all or nothing influence"""
        spoof_0 = self.get_spoof_0()
        pois, auvs, asvs, connectivity_grid, collision_step_size = \
            spoof_0.pois, spoof_0.auvs, spoof_0.asvs, spoof_0.connectivity_grid, spoof_0.collision_step_size
        config = {
            "rewards":
            {
                "influence_heuristic": "line_of_sight",
                "influence_type": "all_or_nothing",
                "trajectory_influence_threshold": 0.0,
                "auv_reward": "difference",
                "asv_reward": "indirect_difference_auv",
                "multi_reward": "single",
                "distance_threshold": 0.0
            }
        }

        rewards = Rewards(
            pois = pois,
            connectivity_grid = connectivity_grid,
            collision_step_size = collision_step_size,
            config = config
        )

        # Just compare the expected output to the actual output
        expected_rewards = [5.6, 1.2]
        expected_G = 10.5
        rewards, G, _ = rewards.compute(auvs=auvs, asvs=asvs)
        self.assertTrue(np.isclose(expected_G, G), "G from rewards.compute() does not match expected G")
        self.assertTrue(np.allclose(expected_rewards, rewards),
            "ASV rewards from rewards.compute() does not match expected rewards")

if __name__ == '__main__':
    unittest.main()
