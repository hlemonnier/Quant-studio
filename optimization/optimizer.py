import itertools
import pandas as pd
import random
import quantstats
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from geneticalgorithm import geneticalgorithm as ga
import numpy as np
import sys
import os
import time
from functools import reduce
import operator


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from parameters.config_file import Config
from parameters.parameters import parameters_details
from backtesting.run_backtest import runstrat


class StrategyOptimizer:
    """
    This class provides methods to optimize trading strategy parameters using different optimization techniques
    including grid search, random search, genetic algorithm, and Bayesian optimization.
    """

    def __init__(self):
        self.Config = Config
        self.runstrat = runstrat


    def timed_optimization(self, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        time_taken = time.time() - start_time
        return result, time_taken


    def evaluate_strategy(self, params):
        """
        Evaluate the trading strategy with given parameters and calculate key performance metrics.

        Parameters:
            params (dict): Dictionary of strategy parameters to be set for the evaluation.

        Returns:
            tuple (float, float): Tuple containing the Sharpe Ratio and Sortino Ratio of the strategy.
        """
        config = self.Config()
        args = config.parse_args()
        
        args.verbose = False
        args.print_logs = False
        args.print_tearsheet = False
        args.plot = False
        args.write = False

        # Set arguments for the strategy run based on the supplied parameters
        for key, value in params.items():
            setattr(args, key, value)

        # Execute the strategy and compute performance metrics
        returns = self.runstrat(args)
        sharpe_ratio = quantstats.stats.sharpe(returns, periods=args.periods, rf=args.riskfree_rate)
        sortino_ratio = quantstats.stats.sortino(returns, periods=args.periods, rf=args.riskfree_rate)
        
        return sharpe_ratio, sortino_ratio
 

    def count_grid_search_iterations(self, params_to_optimize):
        return reduce(operator.mul, (len(values) for values in params_to_optimize.values()), 1)



    def calculate_genetic_algorithm_evaluations(self, n_generations, population_size, elitism_ratio=0.1):
        # All members of the initial generation are evaluated
        initial_evaluations = population_size
        
        # For subsequent generations, only the non-elite children are evaluated
        elite_count = int(population_size * elitism_ratio)
        children_evaluations = (population_size - elite_count) * (n_generations - 1)

        total_evaluations = initial_evaluations + children_evaluations
        return total_evaluations



    def grid_search_optimization(self, params_to_optimize, verbose=True):
        """
        Perform a grid search over all combinations of specified parameter ranges to optimize trading strategy.

        Parameters:
            params_to_optimize (dict): Dictionary where keys are parameter names and values are lists of parameter values to test.

        Returns:
            DataFrame: A pandas DataFrame containing the results of the grid search, sorted by Sharpe Ratio and Sortino Ratio in descending order.
        """
        start_time = time.time()
        iterations = self.count_grid_search_iterations(params_to_optimize)
        parameter_names = ', '.join(params_to_optimize.keys())
        print(f"Beginning a Grid Search optimization on the parameters: {parameter_names}.")

        combinations = itertools.product(*(params_to_optimize[param] for param in params_to_optimize))
        results = []

        for combination in combinations:
            params = dict(zip(params_to_optimize, combination))
            if verbose:
                print(f"Testing parameters: {params}")
            sharpe_ratio, sortino_ratio = self.evaluate_strategy(params)
            results.append({**params, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio})

        df = pd.DataFrame(results).sort_values(['sharpe_ratio', 'sortino_ratio'], ascending=[False, False])
        best_params = df.iloc[0].to_dict()
        end_time = time.time()

        return {
            'params': {key: best_params[key] for key in params_to_optimize.keys()},
            'results': {'sharpe_ratio': best_params['sharpe_ratio'], 'sortino_ratio': best_params['sortino_ratio']},
            'iteration': iterations,
            'time': end_time - start_time
        }



    def random_search_optimization(self, params_to_optimize, n_calls, verbose=True):
        """
        Perform a random search over specified parameter ranges, ensuring unique combinations for each test.

        Parameters:
            params_to_optimize (dict): Dictionary where keys are parameter names and values are lists of values to test.
            n_calls (int): Number of random iterations to perform.

        Returns:
            DataFrame: A pandas DataFrame containing the results of the random search, sorted by Sharpe Ratio and Sortino Ratio in descending order.
        """
        start_time = time.time()
        parameter_names = ', '.join(params_to_optimize.keys())
        print(f"Beginning a Random Search optimization on the parameters: {parameter_names}.")
        results = []
        tested_combinations = set()
        potential_combinations = 1

        for values in params_to_optimize.values():
            potential_combinations *= len(values)
        
        while len(results) < n_calls and len(tested_combinations) < potential_combinations:
            random_params = {param: random.choice(values) for param, values in params_to_optimize.items()}
            params_tuple = tuple(random_params.items())

            if params_tuple not in tested_combinations:
                tested_combinations.add(params_tuple)
                if verbose:
                    print(f"Testing parameters: {random_params}")
                sharpe_ratio, sortino_ratio = self.evaluate_strategy(random_params)
                results.append({**random_params, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio})

        df = pd.DataFrame(results).sort_values(['sharpe_ratio', 'sortino_ratio'], ascending=[False, False])
        best_params = df.iloc[0].to_dict()
        end_time = time.time()

        return {
            'params': {key: best_params[key] for key in params_to_optimize.keys()},
            'results': {'sharpe_ratio': best_params['sharpe_ratio'], 'sortino_ratio': best_params['sortino_ratio']},
            'iteration': n_calls,
            'time': end_time - start_time

        }



    def bayesian_optimization(self, params_to_optimize, n_calls, verbose=True):
        """
        Perform Bayesian optimization to find the best trading strategy parameters.

        Parameters:
            params_to_optimize (dict): Dictionary where keys are parameter names and values are tuples representing the range (min, max).
            n_calls (int): Number of function evaluations to perform.

        Returns:
            DataFrame: A pandas DataFrame containing the results of the Bayesian optimization, sorted by Sharpe Ratio in descending order.
        """
        start_time = time.time()
        parameter_names = ', '.join(params_to_optimize.keys())
        print(f"Beginning a Bayesian optimization on the parameters: {parameter_names}.")

        space = [Integer(low=int(param_range[0]), high=int(param_range[1]), name=param)
                if isinstance(param_range[0], int) else Real(low=param_range[0], high=param_range[1], name=param)
                for param, param_range in params_to_optimize.items()]

        evaluations = []

        @use_named_args(space)
        def objective(**params):
            if verbose:
                print(f"Testing parameters: {params}")
            sharpe_ratio, sortino_ratio = self.evaluate_strategy(params)
            evaluations.append({**params, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio})
            return -sortino_ratio


        gp_minimize(objective, space, n_calls=n_calls, random_state=0)
        df = pd.DataFrame(evaluations).sort_values(['sharpe_ratio', 'sortino_ratio'], ascending=[False, False])
        best_params = df.iloc[0].to_dict()
        end_time = time.time()

        return {
            'params': {key: best_params[key] for key in params_to_optimize.keys()},
            'results': {'sharpe_ratio': best_params['sharpe_ratio'], 'sortino_ratio': best_params['sortino_ratio']},
            'iteration': n_calls,
            'time': end_time - start_time
        }



    def genetic_algorithm_optimization(self, params_to_optimize, n_generations, population_size, verbose=True):
        """
        Perform Genetic Algorithm optimization to find the best trading strategy parameters using the 'geneticalgorithm' package.

        Parameters:
            params_to_optimize (dict): Dictionary where keys are parameter names and values are tuples representing the range (min, max).
            n_generations (int): Number of generations over which the GA will run.
            population_size (int): Number of individuals in each generation.

        Returns:
            dict: Best parameters found and their fitness values.
        """
        start_time = time.time()
        parameter_names = ', '.join(params_to_optimize.keys())
        print(f"Beginning a Genetic Algorithm optimization on the parameters: {parameter_names}.")

        evaluations = []

        def objective(params):
            # Convert the array of parameters to a dictionary with appropriate parameter names
            params_dict = {list(params_to_optimize.keys())[i]: params[i] for i in range(len(params))}
                    
            # Ensure correct types for all parameters
            for param, value in params_dict.items():
                if parameters_details[param]['type'] == int:
                    params_dict[param] = int(round(value))

            if verbose:
                print(f"Testing parameters: {params_dict}")

            sharpe_ratio, sortino_ratio = self.evaluate_strategy(params_dict)
            evaluations.append({**params_dict, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio})

            return -sharpe_ratio  # Minimizing negative Sharpe Ratio as GA objective
                
        varbound = np.array([list(bound) for bound in params_to_optimize.values()])
        vartype = np.array(['int' if parameters_details[param]['type'] == int else 'real' for param in params_to_optimize.keys()])

        algorithm_param = {
            'max_num_iteration': n_generations,
            'population_size': population_size,
            'mutation_probability': 0.1,  # Usually around 0.1
            'elit_ratio': 0.1,  # Smaller elite ratio
            'crossover_probability': 0.5,  # Usually around 0.5
            'parents_portion': 0.3,  # Ensure this is greater than the elit_ratio
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }

        model = ga(function=objective, dimension=len(params_to_optimize), variable_type_mixed=vartype, variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=False, progress_bar=False)
        model.run()

        df = pd.DataFrame(evaluations).sort_values(['sharpe_ratio', 'sortino_ratio'], ascending=[False, False])
        best_params = df.iloc[0].to_dict()
        end_time = time.time()

        return {
            'params': {key: best_params[key] for key in params_to_optimize.keys()},
            'results': {'sharpe_ratio': best_params['sharpe_ratio'], 'sortino_ratio': best_params['sortino_ratio']},
            'iteration': self.calculate_genetic_algorithm_evaluations(algorithm_param['max_num_iteration'], algorithm_param['population_size'], elitism_ratio=algorithm_param['elit_ratio'] ),
            'time': end_time - start_time
            }




