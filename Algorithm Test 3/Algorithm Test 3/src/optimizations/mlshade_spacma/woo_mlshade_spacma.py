"""
Wave Optics Optimizer with mLSHADE-SPACMA Strategies
Integrates mLSHADE-SPACMA strategies into WOO algorithm.
"""

import numpy as np
from src.algorithms.woo_algorithm import WaveOpticsOptimizer
from .mlshade_spacma_adapter import MLSHADESPACMAAdapter


class WOO_MLSHADESPACMA(WaveOpticsOptimizer):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, rho=0.1):
        super().__init__(func, bounds, pop_size, max_generations)
        
        # Initialize mLSHADE-SPACMA adapter
        self.adapter = MLSHADESPACMAAdapter(
            self.func, self.dim, self.pop_size, self.max_generations, self.bounds, rho
        )
        self.adapter.increment_nfes(self.pop_size)
    
    def optimize(self):
        # 1. Initialization
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        # Sort
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        best_solution = population[0].copy()
        best_fitness = fitness[0]
        
        fitness_history = []
        
        # Advantageous Population Params
        MaDim = self.pop_size
        MaDim0 = self.pop_size
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # Apply precise elimination and generation (first half only)
            population, fitness = self.adapter.precise_elimination_and_generation(
                population, fitness
            )
            current_pop_size = len(population)
            
            # Update uncertainty
            self.Ua = 0.5 * (1 - t / self.max_generations)
            
            new_population = np.zeros_like(population)
            successful_F = []
            
            # Select pbest (top p% of population)
            p = 0.11
            p_size = max(1, int(p * current_pop_size))
            pbest_indices = np.random.choice(
                min(p_size, current_pop_size), 
                size=current_pop_size, 
                replace=True
            )
            
            for i in range(current_pop_size):
                X_i = population[i]
                
                # Get adaptive F
                r_i = np.random.randint(1, self.adapter.H + 1)
                F_i = self.adapter.get_adaptive_F(r_i)
                
                # Apply mLSHADE-SPACMA mutation strategy
                X_pbest = population[pbest_indices[i]]
                v_i = self.adapter.apply_mutation_strategy(
                    X_i, X_pbest, population, fitness, F_i
                )
                
                # WOO-specific update (simplified)
                # Wave interference pattern
                idxs = np.random.choice(current_pop_size, 2, replace=False)
                X_r1, X_r2 = population[idxs[0]], population[idxs[1]]
                
                # Phase difference
                phase = np.random.rand(self.dim) * 2 * np.pi
                interference = np.sin(phase) * (X_r1 - X_r2)
                
                X_new = X_i + self.Ua * interference
                
                # Blend with mLSHADE-SPACMA mutation
                alpha = 0.5
                X_new = alpha * v_i + (1 - alpha) * X_new
                
                # Apply bounds
                X_new = np.clip(X_new, self.lb, self.ub)
                
                # Evaluate
                f_new = self.func(X_new)
                self.adapter.increment_nfes(1)
                
                # Selection
                if f_new < fitness[i]:
                    # Update archive
                    self.adapter.update_archive(population[i], X_new, fitness[i], f_new)
                    successful_F.append(F_i)
                    
                    population[i] = X_new
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = X_new.copy()
                
                new_population[i] = population[i]
            
            # Update memory
            self.adapter.update_memory_F(successful_F)
            
            # Re-sort population
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            fitness_history.append(best_fitness)
        
        return best_solution, best_fitness, fitness_history

