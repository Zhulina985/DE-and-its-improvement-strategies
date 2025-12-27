import  numpy as np
import sys
import os
import concurrent.futures
import matplotlib.pyplot as plt
import argparse
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cec2014 import CEC2014
from src.algorithms.de_algorithm import DifferentialEvolution
from src.algorithms.cpo_algorithm import CrestedPorcupineOptimizer
from src.algorithms.flood_algorithm import FloodAlgorithm
from src.algorithms.fgo_algorithm import FungalGrowthOptimizer
from src.algorithms.woo_algorithm import WaveOpticsOptimizer
from src.optimizations.tsms.cpo_tsms import CPO_TSMS
from src.optimizations.tsms.fgo_tsms import FGO_TSMS
from src.optimizations.tsms.flood_tsms import FLOOD_TSMS
from src.optimizations.tsms.woo_tsms import WOO_TSMS
from src.optimizations.dg.cpo_dg import CPO_DG
from src.optimizations.dg.fgo_dg import FGO_DG
from src.optimizations.dg.flood_dg import FLOOD_DG
from src.optimizations.dg.woo_dg import WOO_DG
from src.optimizations.nps.cpo_nps import CPO_NPS
from src.optimizations.nps.fgo_nps import FGO_NPS
from src.optimizations.nps.flood_nps import FLOOD_NPS
from src.optimizations.nps.woo_nps import WOO_NPS
from src.optimizations.hcde.cpo_hcde import CPO_HCDE
from src.optimizations.hcde.fgo_hcde import FGO_HCDE
from src.optimizations.hcde.flood_hcde import FLOOD_HCDE
from src.optimizations.hcde.woo_hcde import WOO_HCDE
from src.optimizations.mlshade_spacma.cpo_mlshade_spacma import CPO_MLSHADESPACMA
from src.optimizations.mlshade_spacma.fgo_mlshade_spacma import FGO_MLSHADESPACMA
from src.optimizations.mlshade_spacma.flood_mlshade_spacma import FLOOD_MLSHADESPACMA
from src.optimizations.mlshade_spacma.woo_mlshade_spacma import WOO_MLSHADESPACMA

def run_single_trial(algo_class, func_num, dim, pop_size, max_generations):
    """
    Runs a single trial of an optimization algorithm.
    """
    # Instantiate inside the process
    # func_instance = func_class(ndim=dim)
    
    # Setup CEC2014 benchmark (Local implementation)
    bench = CEC2014(dim)
    
    # Standard bounds for CEC2014 are [-100, 100]
    lb, ub = -100, 100
    bounds = [(lb, ub)] * dim
        
    def obj_func(x):
        return bench.evaluate(x, func_num)
        
    optimizer = algo_class(
        func=obj_func,
        bounds=bounds,
        pop_size=pop_size,
        max_generations=max_generations
    )
    
    # DE requires extra params, CPO might have defaults. 
    # For simplicity, we assume the class handles its specific params in __init__ defaults 
    # or we could pass **kwargs.
    # Our DE implementation requires extra args in __init__, so we might need to adjust.
    if algo_class == DifferentialEvolution:
         optimizer = algo_class(
            func=obj_func,
            bounds=bounds,
            pop_size=pop_size,
            mutation_factor=0.5,
            crossover_prob=0.9,
            max_generations=max_generations
        )
    
    _, _, history = optimizer.optimize()
    return history

def run_tests_and_plot():
    parser = argparse.ArgumentParser(description='Run optimization algorithms on CEC2014 benchmarks.')
    parser.add_argument('--algos', nargs='+', default=['DE', 'CPO', 'FLOOD', 'FGO', 'WOO'],
                        choices=['DE', 'CPO', 'FLOOD', 'FGO', 'WOO', 'CPO_TSMS', 'FGO_TSMS', 'FLOOD_TSMS', 'WOO_TSMS', 'CPO_DG', 'FGO_DG', 'FLOOD_DG', 'WOO_DG', 'CPO_NPS', 'FGO_NPS', 'FLOOD_NPS', 'WOO_NPS', 'CPO_HCDE', 'FGO_HCDE', 'FLOOD_HCDE', 'WOO_HCDE', 'CPO_MLSHADESPACMA', 'FGO_MLSHADESPACMA', 'FLOOD_MLSHADESPACMA', 'WOO_MLSHADESPACMA'],
                        help='Algorithms to run (default: DE CPO FLOOD FGO WOO)')
    parser.add_argument('--dim', type=int, default=10, help='Dimension of the problem (default: 10)')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size (default: 50)')
    parser.add_argument('--gen', type=int, default=500, help='Max generations (default: 500)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per function (default: 10)')
    
    args = parser.parse_args()
    
    # Configuration
    dim = args.dim
    pop_size = args.pop_size
    max_generations = args.gen
    runs = args.runs
    selected_algos = args.algos
    
    # Updated to use Function ID numbers instead of opfunu classes
    functions_to_test = [


        ("F25",25 )

    ]
    
    # Local CEC2014 instance for theoretical optima lookup
    bench_ref = CEC2014(dim)
    
    all_algorithms = {
        "DE": DifferentialEvolution,
        "CPO": CrestedPorcupineOptimizer,
        "FLOOD": FloodAlgorithm,
        "FGO": FungalGrowthOptimizer,
        "WOO": WaveOpticsOptimizer,
        "CPO_TSMS": CPO_TSMS,
        "FGO_TSMS": FGO_TSMS,
        "FLOOD_TSMS": FLOOD_TSMS,
        "WOO_TSMS": WOO_TSMS,
        "CPO_DG": CPO_DG,
        "FGO_DG": FGO_DG,
        "FLOOD_DG": FLOOD_DG,
        "WOO_DG": WOO_DG,
        "CPO_NPS": CPO_NPS,
        "FGO_NPS": FGO_NPS,
        "FLOOD_NPS": FLOOD_NPS,
        "WOO_NPS": WOO_NPS,
        "CPO_HCDE": CPO_HCDE,
        "FGO_HCDE": FGO_HCDE,
        "FLOOD_HCDE": FLOOD_HCDE,
        "WOO_HCDE": WOO_HCDE,
        "CPO_MLSHADESPACMA": CPO_MLSHADESPACMA,
        "FGO_MLSHADESPACMA": FGO_MLSHADESPACMA,
        "FLOOD_MLSHADESPACMA": FLOOD_MLSHADESPACMA,
        "WOO_MLSHADESPACMA": WOO_MLSHADESPACMA,
    }
    
    algorithms = [(name, all_algorithms[name]) for name in selected_algos]
    
    results = {} # { (algo_name, func_name): history }
    
    # Data storage for CSV
    # Structure: Function, Algorithm, Best_Fitness_Mean, Std_Dev_From_Optimum
    csv_data = []
    
    print(f"Running Comparison on CEC2014 (Local Implementation) (Dim={dim}, Runs={runs})")
    print(f"Algorithms: {selected_algos}")
    print("-" * 60)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for algo_name, algo_class in algorithms:
            for func_name, func_num in functions_to_test:
                print(f"Submitting {runs} runs for {algo_name} on {func_name} (ID: {func_num})...")
                
                futures = [
                    executor.submit(run_single_trial, algo_class, func_num, dim, pop_size, max_generations) 
                    for _ in range(runs)
                ]
                
                histories = []
                final_fitness_values = []
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        history = future.result()
                        histories.append(history)
                        final_fitness_values.append(history[-1])
                    except Exception as e:
                        print(f"Run failed for {algo_name}-{func_name}: {e}")
                
                if histories:
                    histories = np.array(histories)
                    avg_history = np.mean(histories, axis=0)
                    results[(algo_name, func_name)] = avg_history
                    
                    # Calculate stats
                    final_fitness_values = np.array(final_fitness_values)
                    mean_best_fitness = np.mean(final_fitness_values)
                    
                    # Theoretical Optimum
                    # CEC2014 bias values are the optima.
                    # We can get them from bench_ref.bias[func_num]
                    optimum = bench_ref.bias[func_num]
                    
                    # Std Dev from Optimum (or Std Dev of the final fitness values?)
                    # Usually "Std Dev" in tables means std(final_values).
                    # "Std Dev from Optimum" could mean sqrt(mean((x - opt)^2)) (RMSE).
                    # Standard reporting: Mean Error (Mean - Opt) and Std Dev of Error (which is Std Dev of Fitness).
                    # Let's record Mean Best Fitness and Standard Deviation of Best Fitness.
                    
                    std_dev = np.std(final_fitness_values)
                    mean_error = mean_best_fitness - optimum
                    
                    print(f"Finished {algo_name}-{func_name}. Mean Fit: {mean_best_fitness:.6f}, Std: {std_dev:.6f}")
                    
                    csv_data.append({
                        "Function": func_name,
                        "Algorithm": algo_name,
                        "Mean_Best_Fitness": mean_best_fitness,
                        "Std_Dev": std_dev,
                        "Theoretical_Optimum": optimum,
                        "Mean_Error": mean_error
                    })

    # Save to CSV
    df = pd.DataFrame(csv_data)
    
    # Ensure data directory exists
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    output_csv = os.path.join(data_dir, "algorithm_comparison_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Also save a pivot table for quick viewing
    pivot_df = df.pivot(index='Function', columns='Algorithm', values=['Mean_Best_Fitness', 'Std_Dev'])
    pivot_csv = os.path.join(data_dir, "algorithm_comparison_pivot.csv")
    pivot_df.to_csv(pivot_csv)
    print(f"Pivot summary saved to {pivot_csv}")
    
    # Plotting
    print("\nGenerating comparison plots...")
    
    # Create a subplot for each function
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (func_name, _) in enumerate(functions_to_test):
        ax = axes[i]
        
        for algo_name, _ in algorithms:
            key = (algo_name, func_name)
            if key in results:
                history = results[key]
                ax.plot(history, label=algo_name, linewidth=2)
        
        ax.set_yscale('log')
        ax.set_title(f'{func_name} Convergence')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness (Log)')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
    plt.tight_layout()
    
    # Construct filename based on algorithms run
    algo_str = "_".join(selected_algos).lower()
    output_file = os.path.join(data_dir, f'comparison_plot_{algo_str}.png')
    
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    run_tests_and_plot()
