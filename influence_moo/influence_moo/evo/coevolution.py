import multiprocessing
import random
from copy import deepcopy
import pprint
import random
import os
from pathlib import Path

import numpy as np
import yaml

from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm

from influence_moo.evo.network import NeuralNetwork
from influence_moo.env.env import OceanEnv

class JointTrajectory():
    def __init__(
            self, auv_paths, auv_hpaths, auv_actions, auv_crash_histories, auv_surface_histories,
            asv_paths, asv_actions, asv_crash_histories
        ):
        self.auv_paths = auv_paths
        self.auv_hpaths = auv_hpaths
        self.auv_actions = auv_actions
        self.auv_crash_histories = auv_crash_histories
        self.auv_surface_histories = auv_surface_histories

        self.asv_paths = asv_paths
        self.asv_actions = asv_actions
        self.asv_crash_histories = asv_crash_histories

class EvalInfo():
    def __init__(self, rewards, joint_trajectory):
        self.rewards = rewards
        self.joint_trajectory = joint_trajectory

class CooperativeCoevolutionaryAlgorithm():
    def __init__(self, config_dir):
        self.config_dir = Path(os.path.expanduser(config_dir))
        self.trials_dir = self.config_dir.parent

        with open(str(self.config_dir), 'r') as file:
            self.config = yaml.safe_load(file)

        self.num_asvs = self.config['env']['num_asvs']

        self.subpopulation_size = self.config['ccea']['subpopulation_size']
        self.n_elites = self.config["ccea"]["selection"]["n_elites_binary_tournament"]["n_elites"]
        self.include_elites_in_tournament = self.config["ccea"]["selection"]["n_elites_binary_tournament"]["include_elites_in_tournament"]
        self.num_mutants = self.subpopulation_size - self.n_elites
        self.num_evaluations_per_team = self.config["ccea"]["evaluation"]["multi_evaluation"]["num_evaluations_per_team"]

        self.use_multiprocessing = self.config["processing"]["use_multiprocessing"]
        self.num_threads = self.config["processing"]["num_threads"]

        # Data saving variables
        self.save_trajectories = self.config["data"]["save_trajectories"]["switch"]
        self.num_gens_between_save_traj = self.config["data"]["save_trajectories"]["num_gens_between_save"]

        # Create the type of fitness we're optimizing
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Now set up mapping for multiprocessing
        if self.use_multiprocessing:
            self.pool = multiprocessing.Pool(processes=self.num_threads)
            self.map = self.pool.map_async
        else:
            self.map = map

        # Template env for all our rollouts
        # deepcopy this attribute so we always have a clean one
        self.clean_env = OceanEnv(self.config)
        self.num_auvs = len(self.clean_env.mission.paths)
        self.num_pois = len(self.clean_env.mission.pois)

        # For neural network calculations
        self.nn_template = self.generateTemplateNeuralNetwork(num_hidden=self.config['env']['asv']['network']['num_hidden'])

    # This makes it possible to pass evaluation to multiprocessing
    # Without this, the pool tries to pickle the entire object, including itself
    # which it cannot do
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['map']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def generateTemplateNeuralNetwork(self, num_hidden):
        agent_nn = NeuralNetwork(
            num_inputs=2*self.num_asvs+\
                2*len(self.clean_env.mission.paths),
            num_hidden=num_hidden,
            num_outputs=2
        )
        return agent_nn

    def generateWeight(self):
        return random.uniform(self.config["ccea"]["weight_initialization"]["lower_bound"], self.config["ccea"]["weight_initialization"]["upper_bound"])

    def generateIndividual(self, individual_size):
        return tools.initRepeat(creator.Individual, self.generateWeight, n=individual_size)

    def generateAsvIndividual(self):
        return self.generateIndividual(individual_size=self.nn_template.num_weights)

    def generateAsvSubpopulation(self):
        return tools.initRepeat(list, self.generateAsvIndividual, n=self.config["ccea"]["subpopulation_size"])

    def generateUAVSubpopulation(self):
        return tools.initRepeat(list, self.generateUAVIndividual, n=self.config["ccea"]["subpopulation_size"])

    def population(self):
        return tools.initRepeat(list, self.generateAsvSubpopulation, n=self.num_asvs)

    def formEvaluationTeam(self, population):
        eval_team = []
        for subpop in population:
            # Use max with a key function to get the individual with the highest fitness[0] value
            best_ind = max(subpop, key=lambda ind: ind.fitness.values[0])
            eval_team.append(best_ind)
        return eval_team

    def evaluateEvaluationTeam(self, population):
        # Create evaluation team
        eval_team = self.formEvaluationTeam(population)
        # Evaluate that team however many times we are evaluating teams
        eval_teams = [eval_team for _ in range(self.num_evaluations_per_team)]
        return self.evaluateTeams(eval_teams)

    def formTeams(self, population, inds=None):
        # Start a list of teams
        teams = []

        if inds is None:
            team_inds = range(self.subpopulation_size)
        else:
            team_inds = inds

        # For each individual in a subpopulation
        for i in team_inds:
            # Make a team
            team = []
            # For each subpopulation in the population
            for subpop in population:
                # Put the i'th indiviudal on the team
                team.append(subpop[i])
            # Need to save that team for however many evaluations
            # we're doing per team
            for _ in range(self.num_evaluations_per_team):
                # Save that team
                teams.append(team)

        return teams

    def evaluateTeams(self, teams):
        if self.use_multiprocessing:
            jobs = self.map(self.evaluateTeam, teams)
            eval_infos = jobs.get()
        else:
            eval_infos = list(self.map(self.evaluateTeam, teams))
        return eval_infos

    def evaluateTeam(self, team, compute_team_fitness=True):
        # Set up networks
        asv_nns = [deepcopy(self.nn_template) for _ in range(self.num_asvs)]

        # Load in the weights
        for asv_nn, individual in zip(asv_nns, team):
            asv_nn.setWeights(individual)

        asv_policy_functions = [asv_nn.forward for asv_nn in asv_nns]

        # Set up the enviornment
        env = deepcopy(self.clean_env)
        env.run(asv_policy_functions)
        agent_rewards, G = env.rewards.compute(auvs=env.auvs, asvs=env.asvs)
        rewards = tuple([(r,) for r in agent_rewards]+[(G,)])

        return EvalInfo(
            rewards=rewards,
            joint_trajectory=JointTrajectory(
                auv_paths = [auv.path for auv in env.auvs],
                auv_hpaths = [auv.h_path for auv in env.auvs],
                auv_actions = [auv.action_history for auv in env.auvs],
                auv_crash_histories = [auv.crash_history for auv in env.auvs],
                auv_surface_histories = [auv.surface_history for auv in env.auvs],
                asv_paths = [asv.path for asv in env.asvs],
                asv_actions = [asv.action_history for asv in env.asvs],
                asv_crash_histories = [asv.crash_history for asv in env.asvs]
            )
        )

    def mutateIndividual(self, individual):
        return tools.mutGaussian(individual, mu=self.config["ccea"]["mutation"]["mean"], sigma=self.config["ccea"]["mutation"]["std_deviation"], indpb=self.config["ccea"]["mutation"]["independent_probability"])

    def mutate(self, population):
        # Don't mutate the elites from n-elites
        for num_individual in range(self.num_mutants):
            mutant_id = num_individual + self.n_elites
            for subpop in population:
                self.mutateIndividual(subpop[mutant_id])
                del subpop[mutant_id].fitness.values

    def selectSubPopulation(self, subpopulation):
        # Get the best N individuals
        offspring = tools.selBest(subpopulation, self.n_elites)
        if self.include_elites_in_tournament:
            offspring += tools.selTournament(subpopulation, len(subpopulation)-self.n_elites, tournsize=2)
        else:
            # Get the remaining worse individuals
            remaining_offspring = tools.selWorst(subpopulation, len(subpopulation)-self.n_elites)
            # Add those remaining individuals through a binary tournament
            offspring += tools.selTournament(remaining_offspring, len(remaining_offspring), tournsize=2)
        # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
        # that came from the same selected individual
        return [ deepcopy(individual) for individual in offspring ]

    def select(self, population):
        # Offspring is a list of subpopulation
        offspring = []
        # For each subpopulation in the population
        for subpop in population:
            # Perform a selection on that subpopulation and add it to the offspring population
            offspring.append(self.selectSubPopulation(subpop))
        return offspring

    def shuffle(self, population):
        for subpop in population:
            random.shuffle(subpop)

    def assignFitnesses(self, teams, eval_infos):
        # There may be several eval_infos for the same team
        # This is the case if there are many evaluations per team
        # In that case, we need to aggregate those many evaluations into one fitness
        if self.num_evaluations_per_team == 1:
            for team, eval_info in zip(teams, eval_infos):
                fitnesses = eval_info.rewards
                for individual, fit in zip(team, fitnesses):
                    individual.fitness.values = fit
        else:
            team_list = []
            eval_info_list = []
            for team, eval_info in zip(teams, eval_infos):
                team_list.append(team)
                eval_info_list.append(eval_info)

            for team_id, team in enumerate(team_list[::self.num_evaluations_per_team]):
                # Get all the eval infos for this team
                team_eval_infos = eval_info_list[team_id*self.num_evaluations_per_team:(team_id+1)*self.num_evaluations_per_team]
                # Aggregate the fitnesses into a big numpy array
                all_fitnesses = [eval_info.rewards for eval_info in team_eval_infos]
                average_fitnesses = [0 for _ in range(len(all_fitnesses[0]))]
                for fitnesses in all_fitnesses:
                    for count, fit in enumerate(fitnesses):
                        average_fitnesses[count] += fit[0]
                for ind in range(len(average_fitnesses)):
                    average_fitnesses[ind] = average_fitnesses[ind] / self.num_evaluations_per_team
                # And now get that back to the individuals
                fitnesses = tuple([(f,) for f in average_fitnesses])
                for individual, fit in zip(team, fitnesses):
                    individual.fitness.values = fit

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def createEvalFitnessCSV(self, trial_dir):
        eval_fitness_dir = trial_dir / "fitness.csv"
        header = "generation,team_fitness_aggregated"
        for j in range(self.num_asvs):
            header += ",asv_"+str(j)+"_"
        for i in range(self.num_evaluations_per_team):
            header+=",team_fitness_"+str(i)
            for j in range(self.num_asvs):
                header+=",team_"+str(i)+"_asv_"+str(j)
        header += "\n"
        with open(eval_fitness_dir, 'w') as file:
            file.write(header)

    def writeEvalFitnessCSV(self, trial_dir, eval_infos):
        eval_fitness_dir = trial_dir / "fitness.csv"
        gen = str(self.gen)
        if len(eval_infos) == 1:
            eval_info = eval_infos[0]
            team_fit = str(eval_info.rewards[-1][0])
            agent_fits = [str(fit[0]) for fit in eval_info.rewards[:-1]]
            fit_list = [gen, team_fit]+agent_fits
            fit_str = ','.join(fit_list)+'\n'
        else:
            team_eval_infos = []
            for eval_info in eval_infos:
                team_eval_infos.append(eval_info)
            # Aggergate the fitnesses into a big numpy array
            num_ind_per_team = len(team_eval_infos[0].fitnesses)
            all_fit = np.zeros(shape=(self.num_evaluations_per_team, num_ind_per_team))
            for num_eval, eval_info in enumerate(team_eval_infos):
                fitnesses = eval_info.rewards
                for num_ind, fit in enumerate(fitnesses):
                    all_fit[num_eval, num_ind] = fit[0]
                all_fit[num_eval, -1] = fitnesses[-1][0]
            # Now compute a sum/average/min/etc dependending on what config specifies
            agg_fit = np.average(all_fit, axis=0)
            # And now record it all, starting with the aggregated one
            agg_team_fit = str(agg_fit[-1])
            agg_agent_fits = [str(fit) for fit in agg_fit[:-1]]
            fit_str = gen+','+','.join([agg_team_fit]+agg_agent_fits)+','
            # And now add all the fitnesses from individual trials
            # Each row should have the fitnesses for an evaluation
            for row in all_fit:
                team_fit = str(row[-1])
                agent_fits = [str(fit) for fit in row[:-1]]
                fit_str += ','.join([team_fit]+agent_fits)
            fit_str+='\n'
        # Now save it all to the csv
        with open(eval_fitness_dir, 'a') as file:
                file.write(fit_str)

    def writeEvalTrajs(self, trial_dir, eval_infos):
        gen_folder_name = "gen_"+str(self.gen)
        gen_dir = trial_dir / gen_folder_name
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)
        for eval_id, eval_info in enumerate(eval_infos):
            eval_filename = "eval_team_"+str(eval_id)+"_joint_traj.csv"
            eval_dir = gen_dir / eval_filename
            with open(eval_dir, 'w') as file:
                # Build up the header (labels at the top of the csv)
                header = ""
                # First the states (auvs asvs pois)
                for i in range(self.num_auvs):
                    header += "auv_"+str(i)+"_x,auv_"+str(i)+"_y,"
                for i in range(self.num_auvs):
                    header += "auv_"+str(i)+"_hx,auv_"+str(i)+"_hy,"
                for i in range(self.num_asvs):
                    header += "asv_"+str(i)+"_x,asv_"+str(i)+"_y,"
                for i in range(self.num_pois):
                    header += "poi_"+str(i)+"_x,poi_"+str(i)+"_y,"
                # NO Observations. Observations are just a reorganized subset of states
                # Actions
                for i in range(self.num_auvs):
                    header += "auv_"+str(i)+"_dx,auv_"+str(i)+"_dy,"
                for i in range(self.num_asvs):
                    header += "asv_"+str(i)+"_dx,asv_"+str(i)+"_dy,"
                header+="\n"
                # Write out the header at the top of the csv
                file.write(header)
                # Now fill in the csv with the data
                # One line at a time
                joint_traj = eval_info.joint_trajectory
                """ We're going to pad the actions with np.nan because
                the agents cannot take actions at the last timestep, but
                there is a final joint state """
                # Pad auv actions
                for auv_action_history in joint_traj.auv_actions:
                    auv_action_history.append([np.nan for _ in auv_action_history[0]])
                # Pad asv actions
                for asv_action_history in joint_traj.asv_actions:
                    asv_action_history.append([np.nan for _ in asv_action_history[0]])
                # Transform lists into some way we can save them line by line
                # Start with states
                joint_state_history = [[] for _ in joint_traj.auv_paths[0]]
                for auv_path in joint_traj.auv_paths:
                    for t, auv_state in enumerate(auv_path):
                        joint_state_history[t] += [s for s in auv_state]
                for auv_hpath in joint_traj.auv_hpaths:
                    for t, auv_state in enumerate(auv_hpath):
                        joint_state_history[t] += [s for s in auv_state]
                for asv_path in joint_traj.asv_paths:
                    for t, asv_state in enumerate(asv_path):
                        joint_state_history[t] += [s for s in asv_state]
                for t in range(len(joint_state_history)):
                    for poi in self.clean_env.pois:
                        joint_state_history[t] += [poi.position[0], poi.position[1]]
                # Then actions
                joint_action_history = [[] for _ in joint_traj.auv_actions[0]]
                for auv_action_history in joint_traj.auv_actions:
                    for t, auv_action in enumerate(auv_action_history):
                        joint_action_history[t] += [a for a in auv_action]
                for asv_action_history in joint_traj.asv_actions:
                    for t, asv_action in enumerate(asv_action_history):
                        joint_action_history[t] += [a for a in asv_action]
                for joint_state, joint_action in zip(joint_state_history, joint_action_history):
                    # Aggregate state info
                    state_list = []
                    for state in joint_state:
                        state_list+=[str(state)]
                    state_str = ','.join(state_list)
                    # Aggregate action info
                    action_list = []
                    for action in joint_action:
                        action_list+=[str(action)]
                    action_str = ','.join(action_list)
                    # Put it all together
                    csv_line = state_str+','+action_str+'\n'
                    # Write it out
                    file.write(csv_line)

    def runTrial(self, num_trial):
        # Init gen counter
        self.gen = 0

        # Create directory for saving data
        trial_dir = self.trials_dir / ("trial_"+str(num_trial))
        if not os.path.isdir(trial_dir):
            os.makedirs(trial_dir)

        # Create csv file for saving evaluation fitnesses
        self.createEvalFitnessCSV(trial_dir)

        # Initialize the population
        pop = self.population()

        # Create the teams
        teams = self.formTeams(pop)

        # Evaluate the teams
        eval_infos = self.evaluateTeams(teams)

        # Assign fitnesses to individuals
        self.assignFitnesses(teams, eval_infos)

        # Evaluate a team with the best indivdiual from each subpopulation
        eval_infos = self.evaluateEvaluationTeam(pop)

        # Save fitnesses of the evaluation team
        self.writeEvalFitnessCSV(trial_dir, eval_infos)

        # Save trajectories of evaluation team
        if self.save_trajectories:
            self.writeEvalTrajs(trial_dir, eval_infos)

        for gen in tqdm(range(self.config["ccea"]["num_generations"])):
            # Update gen counter
            self.gen = gen+1

            # Perform selection
            offspring = self.select(pop)

            # Perform mutation
            self.mutate(offspring)

            # Shuffle subpopulations in offspring
            # to make teams random
            self.shuffle(offspring)

            # Form teams for evaluation
            teams = self.formTeams(offspring)

            # Evaluate each team
            eval_infos = self.evaluateTeams(teams)

            # Now assign fitnesses to each individual
            self.assignFitnesses(teams, eval_infos)

            # Evaluate a team with the best indivdiual from each subpopulation
            eval_infos = self.evaluateEvaluationTeam(offspring)

            # Save fitnesses
            self.writeEvalFitnessCSV(trial_dir, eval_infos)

            # Save trajectories
            if self.save_trajectories and self.gen % self.num_gens_between_save_traj == 0:
                self.writeEvalTrajs(trial_dir, eval_infos)

            # Now populate the population with individuals from the offspring
            self.setPopulation(pop, offspring)

    def run(self, num_trial):
        if num_trial is None:
            # Run all trials if no number is specified
            for num_trial in range(self.config["experiment"]["num_trials"]):
                self.runTrial(num_trial)
        else:
            # Run only the trial specified
            self.runTrial(num_trial)

        if self.use_multiprocessing:
            self.pool.close()

def runCCEA(config_dir, num_trial=None):
    ccea = CooperativeCoevolutionaryAlgorithm(config_dir)
    return ccea.run(num_trial)
