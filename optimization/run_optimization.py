from optimizer import StrategyOptimizer
import matplotlib.pyplot as plt

if __name__ == '__main__':

    optimizer = StrategyOptimizer()

    discrete_param_options = {
        "max_risk_per_trade": (0.01, 0.015, 0.018),        
        "ema_period1": (25, 28, 31, 35, 40),
        "sma_period" : (27, 30, 33, 35, 40, 45)
        }

    continuous_param_ranges = {
        "atr_long_multiplier": (1.2, 2.5),
        "atr_short_multiplier": (1.2, 2.5),
        "scale_out_factor": (1/9, 2/3),
        "scale_in_factor": (1/8, 2/3),
        "position_scaling_hold": (2, 5),
        "max_position_count": (2, 5),
        "sizers_percent": (15, 40),
}

    # Perform grid search - random search - bayesian - genetic

    #grid_search_results = optimizer.grid_search_optimization(discrete_param_options, verbose=True)
    #random_search_results = optimizer.random_search_optimization(discrete_param_options, n_calls=4, verbose=True)
    bayesian_optimization_results = optimizer.bayesian_optimization(continuous_param_ranges, n_calls=35, verbose=True)
    #genetic_algorithm_results = optimizer.genetic_algorithm_optimization(continuous_param_ranges, n_generations=3, population_size=5, verbose=True)
    
    print("Optimized Parameters and Performance:")
    #print(grid_search_results)
    #print(random_search_results)
    print(bayesian_optimization_results)
    #print(genetic_algorithm_results)


    
'''    # Optimization Algorithm Comparison Plot
    
    random_search_df = {"iterations": [], "sharpe_ratio":[] }
    bayesian_optimization_df = {"iterations": [], "sharpe_ratio":[] }
    #genetic_algorithm_df = {"iterations": [], "sharpe_ratio":[] }


    iterations = [10,12,14,16,18,20,25,28,30,35]
    for n_calls in iterations:
        random_search_results = optimizer.random_search_optimization(discrete_param_options, n_calls=n_calls, verbose=False)
        random_search_df["iterations"].append(random_search_results["iteration"])
        random_search_df["sharpe_ratio"].append(random_search_results["results"]["sharpe_ratio"])
        bayesian_results = optimizer.bayesian_optimization(continuous_param_ranges, n_calls=n_calls, verbose=False)
        bayesian_optimization_df["iterations"].append(bayesian_results["iteration"])
        bayesian_optimization_df["sharpe_ratio"].append(bayesian_results["results"]["sharpe_ratio"])


    # Define different combinations of n_generations and population_size to test
    genetic_params = [
        {'n_generations': 1, 'population_size': 15},
        {'n_generations': 2, 'population_size': 10},
        {'n_generations': 2, 'population_size': 13},
        {'n_generations': 3, 'population_size': 10},
        {'n_generations': 3, 'population_size': 13},
        {'n_generations': 4, 'population_size': 10}
    ]

    # Iterate over each combination
    for params in genetic_params:
        genetic_results = optimizer.genetic_algorithm_optimization(continuous_param_ranges, **params, verbose=False)
        genetic_algorithm_df["iterations"].append(genetic_results["iteration"])
        genetic_algorithm_df["sharpe_ratio"].append(genetic_results["results"]["sharpe_ratio"])


    # Plotting the results
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines for each method
    ax.plot(random_search_df["iterations"], random_search_df["sharpe_ratio"], label='Random Search')
    ax.plot(bayesian_optimization_df["iterations"], bayesian_optimization_df["sharpe_ratio"], label='Bayesian Optimization')
    #ax.plot(genetic_algorithm_df["iterations"], genetic_algorithm_df["sharpe_ratio"], label='Genetic Algorithm')

    # Add labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Optimization Algorithm Performance Comparison')
    ax.legend()

    plt.grid(True)
    plt.show()'''
    
    