"""
Comparison script for HCDE-enhanced algorithms
Compares original algorithms with NPS and HCDE versions
"""

import numpy as np
from cec2014 import CEC2014
from cpo_algorithm import CrestedPorcupineOptimizer
from fgo_algorithm import FungalGrowthOptimizer
from flood_algorithm import FloodAlgorithm
from woo_algorithm import WaveOpticsOptimizer
from cpo_nps import CPO_NPS
from fgo_nps import FGO_NPS
from flood_nps import FLOOD_NPS
from woo_nps import WOO_NPS
from cpo_hcde import CPO_HCDE
from fgo_hcde import FGO_HCDE
from flood_hcde import FLOOD_HCDE
from woo_hcde import WOO_HCDE
import pandas as pd
import argparse


def run_algorithm(algo_class, func_num, dim, pop_size, max_generations, runs=10):
    """
    Run an algorithm multiple times and collect statistics
    """
    bench = CEC2014(dim, data_dir="CEC2014_input_data")
    lb, ub = -100, 100
    bounds = [(lb, ub)] * dim
    
    def obj_func(x):
        return bench.evaluate(x, func_num)
    
    results = []
    
    for run in range(runs):
        optimizer = algo_class(
            func=obj_func,
            bounds=bounds,
            pop_size=pop_size,
            max_generations=max_generations
        )
        
        _, best_fitness, _ = optimizer.optimize()
        optimum = bench.bias[func_num]
        error = best_fitness - optimum
        results.append(error)
    
    results = np.array(results)
    mean_error = np.mean(results)
    std_error = np.std(results)
    
    return mean_error, std_error, results


def compare_algorithms():
    """
    Compare original, NPS, and HCDE versions of algorithms
    """
    parser = argparse.ArgumentParser(description='Compare HCDE-enhanced algorithms')
    parser.add_argument('--dim', type=int, default=10, help='Dimension (default: 10)')
    parser.add_argument('--pop_size', type=int, default=50, help='Population size (default: 50)')
    parser.add_argument('--gen', type=int, default=500, help='Max generations (default: 500)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs (default: 10)')
    parser.add_argument('--functions', nargs='+', type=int, default=[1, 4, 17, 23],
                        help='Function numbers to test (default: 1 4 17 23)')
    
    args = parser.parse_args()
    
    # Algorithm groups
    algorithm_groups = {
        'CPO': {
            'Original': CrestedPorcupineOptimizer,
            'NPS': CPO_NPS,
            'HCDE': CPO_HCDE
        },
        'FGO': {
            'Original': FungalGrowthOptimizer,
            'NPS': FGO_NPS,
            'HCDE': FGO_HCDE
        },
        'FLOOD': {
            'Original': FloodAlgorithm,
            'NPS': FLOOD_NPS,
            'HCDE': FLOOD_HCDE
        },
        'WOO': {
            'Original': WaveOpticsOptimizer,
            'NPS': WOO_NPS,
            'HCDE': WOO_HCDE
        }
    }
    
    # Collect results
    all_results = []
    
    print("=" * 80)
    print(f"Comparing Algorithms with HCDE Strategies")
    print(f"Dimension: {args.dim}, Population Size: {args.pop_size}, Generations: {args.gen}")
    print(f"Functions: {args.functions}, Runs per test: {args.runs}")
    print("=" * 80)
    
    for algo_name, variants in algorithm_groups.items():
        print(f"\n{algo_name} Algorithm Comparison:")
        print("-" * 80)
        
        for func_num in args.functions:
            print(f"\n  Function F{func_num}:")
            
            for variant_name, algo_class in variants.items():
                try:
                    mean_err, std_err, _ = run_algorithm(
                        algo_class, func_num, args.dim, 
                        args.pop_size, args.gen, args.runs
                    )
                    
                    all_results.append({
                        'Algorithm': algo_name,
                        'Variant': variant_name,
                        'Function': f'F{func_num}',
                        'Mean_Error': mean_err,
                        'Std_Error': std_err
                    })
                    
                    print(f"    {variant_name:10s}: Mean={mean_err:.6e}, Std={std_err:.6e}")
                    
                except Exception as e:
                    print(f"    {variant_name:10s}: Error - {e}")
                    all_results.append({
                        'Algorithm': algo_name,
                        'Variant': variant_name,
                        'Function': f'F{func_num}',
                        'Mean_Error': np.nan,
                        'Std_Error': np.nan
                    })
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    output_file = 'hcde_comparison_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to {output_file}")
    
    # Create summary pivot table
    pivot_mean = df.pivot_table(
        index=['Algorithm', 'Function'], 
        columns='Variant', 
        values='Mean_Error'
    )
    pivot_std = df.pivot_table(
        index=['Algorithm', 'Function'], 
        columns='Variant', 
        values='Std_Error'
    )
    
    summary_file = 'hcde_comparison_summary.csv'
    with open(summary_file, 'w') as f:
        f.write("Mean Error Summary:\n")
        f.write(pivot_mean.to_string())
        f.write("\n\nStandard Deviation Summary:\n")
        f.write(pivot_std.to_string())
    
    print(f"Summary saved to {summary_file}")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    compare_algorithms()

