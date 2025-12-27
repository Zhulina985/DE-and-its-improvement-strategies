# Algorithm Test 3 - Optimization Algorithm Benchmark Framework

## Project Overview

This project is a comprehensive benchmark testing framework for optimization algorithms, focused on evaluating and comparing various metaheuristic optimization algorithms and their enhanced versions on the CEC2014 test suite. The project implements multiple base optimization algorithms and integrates various optimization strategies to enhance them, providing complete testing, comparison, and visualization capabilities.

## Key Features

- 🎯 **Complete CEC2014 Test Suite Implementation**: Includes 30 standard test functions (F1-F30)
- 🔬 **Multiple Base Optimization Algorithms**: Implements 5 mainstream metaheuristic algorithms
- ⚡ **Multiple Optimization Strategy Enhancements**: 5 different optimization strategies applied to base algorithms
- 📊 **Comprehensive Performance Evaluation**: Automatically generates comparison reports, convergence curves, and statistical results
- 🚀 **Parallel Computing Support**: Uses multiprocessing to accelerate algorithm testing
- 📈 **Visualization Analysis**: Automatically generates convergence curves and comparison charts

## Project Structure

```
Algorithm Test 3/
├── src/                          # Source code directory
│   ├── cec2014.py               # CEC2014 test suite implementation
│   ├── algorithms/              # Base optimization algorithms
│   │   ├── de_algorithm.py     # Differential Evolution (DE)
│   │   ├── cpo_algorithm.py    # Crested Porcupine Optimizer (CPO)
│   │   ├── flood_algorithm.py  # Flood Algorithm (FLOOD)
│   │   ├── fgo_algorithm.py    # Fungal Growth Optimizer (FGO)
│   │   └── woo_algorithm.py    # Wave Optics Optimizer (WOO)
│   └── optimizations/          # Optimization strategy enhancement modules
│       ├── dg/                 # Deep Guidance strategy
│       ├── hcde/               # HCDE (Hierarchically Controlled Differential Evolution)
│       ├── mlshade_spacma/     # mLSHADE-SPACMA strategy
│       ├── nps/                # DE-NPS strategies
│       └── tsms/               # TSMS (Triple Strategy Mutation Strategy)
├── tests/                       # Test scripts
│   ├── test_cec2014.py         # Main testing and comparison script
│   ├── compare_hcde_strategies.py    # HCDE strategy comparison
│   └── compare_mlshade_spacma_strategies.py  # mLSHADE-SPACMA strategy comparison
├── data/                        # Data and results directory
│   ├── CEC2014_input_data/     # CEC2014 test data files
│   ├── *.csv                   # Algorithm comparison results (CSV format)
│   └── *.png                   # Convergence curves and comparison charts
└── README.md                   # Project documentation
```

## Implemented Algorithms

### Base Optimization Algorithms

1. **DE (Differential Evolution)** - Differential Evolution Algorithm
   - Classic differential evolution implementation
   - Supports customizable mutation factor and crossover probability

2. **CPO (Crested Porcupine Optimizer)** - Crested Porcupine Optimizer
   - Optimization algorithm based on crested porcupine foraging behavior
   - Includes cyclical population reduction mechanism

3. **FLOOD (Flood Algorithm)** - Flood Algorithm
   - Optimization algorithm based on flood behavior
   - Features exploration and exploitation balance mechanism

4. **FGO (Fungal Growth Optimizer)** - Fungal Growth Optimizer
   - Optimization algorithm simulating fungal growth process
   - Adaptive exploration probability parameters

5. **WOO (Wave Optics Optimizer)** - Wave Optics Optimizer
   - Optimization algorithm based on wave optics principles
   - Utilizes wave interference and diffraction mechanisms

### Optimization Strategy Enhancements

Each base algorithm has 5 enhanced versions using the following strategies:

1. **TSMS (Triple Strategy Mutation Strategy)** - Triple Strategy Mutation Strategy
   - Includes three different mutation strategies
   - Uses opposition-based learning initialization

2. **DG (Deep Guidance)** - Deep Guidance Strategy
   - Uses neural networks to predict optimal directions
   - Multi-layer perceptron integrated with Adam optimizer

3. **NPS (DE-NPS Strategies)** - DE-NPS Strategies
   - Enhanced differential evolution strategies
   - Improved population management mechanism

4. **HCDE (Hierarchically Controlled Differential Evolution)** - Hierarchically Controlled Differential Evolution
   - Entropy-based diversity measurement
   - Hybrid perturbation (Gaussian + Cauchy distribution)
   - Multi-level archive mechanism
   - Bi-stage parameter adaptation

5. **mLSHADE-SPACMA** - Multi-LSHADE combined with SPACMA
   - Linear population size reduction
   - Adaptive parameter control
   - Archive mechanism

## CEC2014 Test Suite

The project implements the complete CEC2014 test suite with 30 test functions:

- **F1-F16**: Basic functions (rotation and shift transformations)
- **F17-F22**: Hybrid functions (combinations of multiple basic functions)
- **F23-F30**: Composition functions (weighted combinations of multiple functions)

Supported dimensions: 2, 10, 20, 30, 50, 100

## Installation and Usage

### Requirements

```bash
numpy
scipy
matplotlib
pandas
```

### Basic Usage

#### 1. Run Algorithm Comparison Tests

```bash
cd "Algorithm Test 3"
python tests/test_cec2014.py --algos DE CPO FLOOD --dim 10 --gen 500 --runs 10
```

Parameters:
- `--algos`: List of algorithms to test (options: DE, CPO, FLOOD, FGO, WOO and their enhanced versions)
- `--dim`: Problem dimension (default: 10)
- `--pop_size`: Population size (default: 50)
- `--gen`: Maximum number of generations (default: 500)
- `--runs`: Number of runs per algorithm (default: 10)

#### 2. Compare HCDE Enhancement Strategies

```bash
python tests/compare_hcde_strategies.py --dim 10 --gen 500 --runs 10 --functions 1 4 17 23
```

#### 3. Compare mLSHADE-SPACMA Strategies

```bash
python tests/compare_mlshade_spacma_strategies.py --dim 10 --gen 500 --runs 10 --functions 1 4 17 23
```

### Available Algorithm Identifiers

**Base Algorithms:**
- `DE` - Differential Evolution
- `CPO` - Crested Porcupine Optimizer
- `FLOOD` - Flood Algorithm
- `FGO` - Fungal Growth Optimizer
- `WOO` - Wave Optics Optimizer

**TSMS Enhanced Versions:**
- `CPO_TSMS`, `FGO_TSMS`, `FLOOD_TSMS`, `WOO_TSMS`

**DG Enhanced Versions:**
- `CPO_DG`, `FGO_DG`, `FLOOD_DG`, `WOO_DG`

**NPS Enhanced Versions:**
- `CPO_NPS`, `FGO_NPS`, `FLOOD_NPS`, `WOO_NPS`

**HCDE Enhanced Versions:**
- `CPO_HCDE`, `FGO_HCDE`, `FLOOD_HCDE`, `WOO_HCDE`

**mLSHADE-SPACMA Enhanced Versions:**
- `CPO_MLSHADESPACMA`, `FGO_MLSHADESPACMA`, `FLOOD_MLSHADESPACMA`, `WOO_MLSHADESPACMA`

## Output Results

After testing, results are saved in the `data/` directory:

1. **CSV Result Files**:
   - `algorithm_comparison_results.csv` - Detailed algorithm comparison results
   - `algorithm_comparison_pivot.csv` - Summary results in pivot table format

2. **Visualization Charts**:
   - `comparison_plot_*.png` - Algorithm convergence curve comparison charts
   - `convergence_plot_F*.png` - Convergence curves for individual functions

The CSV files contain the following fields:
- `Function`: Test function name (F1-F30)
- `Algorithm`: Algorithm name
- `Mean_Best_Fitness`: Mean best fitness value
- `Std_Dev`: Standard deviation
- `Theoretical_Optimum`: Theoretical optimum value (CEC2014 bias value)
- `Mean_Error`: Mean error (Mean best fitness - Theoretical optimum)

## Usage Examples

### Example 1: Compare Base Algorithms

```bash
python tests/test_cec2014.py --algos DE CPO FLOOD FGO WOO --dim 20 --gen 1000 --runs 30
```

### Example 2: Test TSMS Enhanced Versions

```bash
python tests/test_cec2014.py --algos CPO CPO_TSMS FGO FGO_TSMS --dim 10 --gen 500 --runs 20
```

### Example 3: Comprehensive Comparison of All Algorithms

```bash
python tests/test_cec2014.py --algos DE CPO CPO_TSMS CPO_DG CPO_NPS CPO_HCDE CPO_MLSHADESPACMA --dim 10 --gen 500 --runs 10
```

## Code Structure

### Algorithm Interface

All algorithms follow a unified interface:

```python
class Optimizer:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        """
        Args:
            func: Objective function (minimization)
            bounds: Bounds list [(min, max), ...]
            pop_size: Population size
            max_generations: Maximum number of generations
        """
        
    def optimize(self):
        """
        Returns:
            best_solution: Best solution
            best_fitness: Best fitness value
            fitness_history: Fitness history record
        """
```

### CEC2014 Interface

```python
from src.cec2014 import CEC2014

bench = CEC2014(dim=10)  # Create 10-dimensional test suite
fitness = bench.evaluate(solution, func_num=1)  # Evaluate F1 function
```

## Performance Optimization

- Uses multiprocessing for parallel execution of multiple independent runs
- Optimized NumPy operations
- Efficient data structures and algorithm implementations

## Notes

1. **Data Files**: Ensure the `data/CEC2014_input_data/` directory contains all required CEC2014 test data files
2. **Memory Usage**: Large-scale tests (high dimensions, large populations) may consume significant memory
3. **Runtime**: Complete testing of all algorithms and functions may take considerable time; it is recommended to start with small-scale tests

## Extension Development

### Adding New Algorithms

1. Create a new algorithm class in `src/algorithms/`
2. Implement the unified `optimize()` interface
3. Register the new algorithm in the test scripts

### Adding New Optimization Strategies

1. Create a new strategy directory in `src/optimizations/`
2. Implement the strategy adapter class
3. Create enhanced version classes for base algorithms
4. Update test scripts to support the new strategy

## References

- CEC2014 Test Suite: IEEE CEC 2014 Competition on Single Objective Real-Parameter Numerical Optimization
- Original papers for various algorithms and strategies


## Authors

-Zhang Letian
-Zheng Peiyang
-Zhu Junyu


