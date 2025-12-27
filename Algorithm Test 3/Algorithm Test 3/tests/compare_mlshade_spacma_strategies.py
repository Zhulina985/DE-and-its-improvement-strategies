"""
Comparison script for mLSHADE-SPACMA enhanced algorithms
Compares original algorithms with their mLSHADE-SPACMA enhanced versions
"""

import numpy as np
from cec2014 import CEC2014
from cpo_algorithm import CrestedPorcupineOptimizer
from fgo_algorithm import FungalGrowthOptimizer
from flood_algorithm import FloodAlgorithm
from woo_algorithm import WaveOpticsOptimizer
from cpo_mlshade_spacma import CPO_MLSHADESPACMA
from fgo_mlshade_spacma import FGO_MLSHADESPACMA
from flood_mlshade_spacma import FLOOD_MLSHADESPACMA
from woo_mlshade_spacma import WOO_MLSHADESPACMA
import concurrent.futures
import matplotlib.pyplot as plt
import argparse
import pandas as pd


def run_single_trial(algo_class, func_num, dim, pop_size, max_generations):
    """Run a single trial of an optimization algorithm."""
    bench = CEC2014(dim, data_dir="CEC2014_input_data")
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
    
    _, _, history = optimizer.optimize()
    return history


def main():
    parser = argparse.ArgumentParser(
        description='Compare original algorithms with mLSHADE-SPACMA enhanced versions'
    )
    parser.add_argument('--dim', type=int, default=10, help='Dimension (default: 10)')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size (default: 50)')
    parser.add_argument('--gen', type=int, default=500, help='Max generations (default: 500)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs (default: 10)')
    parser.add_argument('--functions', nargs='+', type=int, default=[1, 4, 17, 23],
                       help='Function numbers to test (default: 1 4 17 23)')
    
    args = parser.parse_args()
    
    # Algorithm pairs: (original, enhanced)
    algorithm_pairs = [
        ("CPO", CrestedPorcupineOptimizer, "CPO_MLSHADESPACMA", CPO_MLSHADESPACMA),
        ("FGO", FungalGrowthOptimizer, "FGO_MLSHADESPACMA", FGO_MLSHADESPACMA),
        ("FLOOD", FloodAlgorithm, "FLOOD_MLSHADESPACMA", FLOOD_MLSHADESPACMA),
        ("WOO", WaveOpticsOptimizer, "WOO_MLSHADESPACMA", WOO_MLSHADESPACMA),
    ]
    
    functions_to_test = [(f"F{num}", num) for num in args.functions]
    bench_ref = CEC2014(args.dim, data_dir="CEC2014_input_data")
    
    results = {}  # {(algo_name, func_name): history}
    csv_data = []
    
    print(f"Comparing mLSHADE-SPACMA Enhanced Algorithms")
    print(f"Dimension: {args.dim}, Runs: {args.runs}, Generations: {args.gen}")
    print("-" * 80)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for orig_name, orig_class, enh_name, enh_class in algorithm_pairs:
            for func_name, func_num in functions_to_test:
                # Test original
                print(f"Testing {orig_name} on {func_name}...")
                futures_orig = [
                    executor.submit(run_single_trial, orig_class, func_num, 
                                   args.dim, args.pop_size, args.gen)
                    for _ in range(args.runs)
                ]
                
                histories_orig = []
                final_fitness_orig = []
                
                for future in concurrent.futures.as_completed(futures_orig):
                    try:
                        history = future.result()
                        histories_orig.append(history)
                        final_fitness_orig.append(history[-1])
                    except Exception as e:
                        print(f"  Error in {orig_name}-{func_name}: {e}")
                
                # Test enhanced
                print(f"Testing {enh_name} on {func_name}...")
                futures_enh = [
                    executor.submit(run_single_trial, enh_class, func_num,
                                   args.dim, args.pop_size, args.gen)
                    for _ in range(args.runs)
                ]
                
                histories_enh = []
                final_fitness_enh = []
                
                for future in concurrent.futures.as_completed(futures_enh):
                    try:
                        history = future.result()
                        histories_enh.append(history)
                        final_fitness_enh.append(history[-1])
                    except Exception as e:
                        print(f"  Error in {enh_name}-{func_name}: {e}")
                
                # Process results
                if histories_orig and histories_enh:
                    histories_orig = np.array(histories_orig)
                    histories_enh = np.array(histories_enh)
                    
                    avg_history_orig = np.mean(histories_orig, axis=0)
                    avg_history_enh = np.mean(histories_enh, axis=0)
                    
                    results[(orig_name, func_name)] = avg_history_orig
                    results[(enh_name, func_name)] = avg_history_enh
                    
                    # Statistics
                    final_orig = np.array(final_fitness_orig)
                    final_enh = np.array(final_fitness_enh)
                    
                    mean_orig = np.mean(final_orig)
                    std_orig = np.std(final_orig)
                    mean_enh = np.mean(final_enh)
                    std_enh = np.std(final_enh)
                    
                    optimum = bench_ref.bias[func_num]
                    improvement = mean_orig - mean_enh  # Positive means enhanced is better
                    improvement_pct = (improvement / (mean_orig - optimum + 1e-10)) * 100
                    
                    print(f"  {orig_name}: Mean={mean_orig:.6f}, Std={std_orig:.6f}")
                    print(f"  {enh_name}: Mean={mean_enh:.6f}, Std={std_enh:.6f}")
                    print(f"  Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
                    print()
                    
                    # CSV data
                    csv_data.append({
                        "Function": func_name,
                        "Algorithm": orig_name,
                        "Mean_Best_Fitness": mean_orig,
                        "Std_Dev": std_orig,
                        "Theoretical_Optimum": optimum,
                        "Mean_Error": mean_orig - optimum
                    })
                    csv_data.append({
                        "Function": func_name,
                        "Algorithm": enh_name,
                        "Mean_Best_Fitness": mean_enh,
                        "Std_Dev": std_enh,
                        "Theoretical_Optimum": optimum,
                        "Mean_Error": mean_enh - optimum
                    })
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    output_csv = "mlshade_spacma_comparison_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    
    num_functions = len(functions_to_test)
    fig, axes = plt.subplots(num_functions, 1, figsize=(12, 4 * num_functions))
    if num_functions == 1:
        axes = [axes]
    
    for i, (func_name, _) in enumerate(functions_to_test):
        ax = axes[i]
        
        for orig_name, _, enh_name, _ in algorithm_pairs:
            # Plot original
            key_orig = (orig_name, func_name)
            if key_orig in results:
                history = results[key_orig]
                ax.plot(history, label=orig_name, linewidth=2, linestyle='--', alpha=0.7)
            
            # Plot enhanced
            key_enh = (enh_name, func_name)
            if key_enh in results:
                history = results[key_enh]
                ax.plot(history, label=enh_name, linewidth=2, linestyle='-')
        
        ax.set_yscale('log')
        ax.set_title(f'{func_name} Convergence Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness (Log Scale)')
        ax.legend(loc='upper right')
        ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    output_file = 'mlshade_spacma_comparison_plot.png'
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")
    
    print("\nComparison completed!")


if __name__ == "__main__":
    main()

