"""
Flood Algorithm with mLSHADE-SPACMA Strategies
Integrates mLSHADE-SPACMA strategies into FLOOD algorithm.
"""

import numpy as np
from src.algorithms.flood_algorithm import FloodAlgorithm
from .mlshade_spacma_adapter import MLSHADESPACMAAdapter


class FLOOD_MLSHADESPACMA(FloodAlgorithm):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, Ne=5, rho=0.1):
        super().__init__(func, bounds, pop_size, max_generations, Ne)
        
        # Initialize mLSHADE-SPACMA adapter
        self.adapter = MLSHADESPACMAAdapter(
            self.func, self.dim, self.pop_size, self.max_generations, self.bounds, rho
        )
        self.adapter.increment_nfes(self.pop_size)
    
    def optimize(self):
        # 1. Initialization
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        # Find global best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # Apply precise elimination and generation (first half only)
            population, fitness = self.adapter.precise_elimination_and_generation(
                population, fitness
            )
            current_pop_size = len(population)
            
            # Update global stats
            f_min = np.min(fitness)
            f_max = np.max(fitness)
            denom = f_max - f_min
            if denom == 0: denom = 1e-10
            
            # Calculate Pk (Water Loss Coefficient) for Flood Disturbance
            T = self.max_generations
            term_A = np.sqrt(T * (t**2) + 1)
            term_B = np.log(term_A + T/4.0)
            term_denom = (T/4.0) * t
            bracket = (term_A * term_B) / term_denom
            if bracket <= 0: bracket = 1e-10
            Pk = (1.2 / t) * (bracket ** (-2.0/3.0))
            
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
                
                # FLOOD-specific update
                if np.random.rand() < Pk:
                    # Flood Disturbance
                    idxs = np.random.choice(current_pop_size, 2, replace=False)
                    X_r1, X_r2 = population[idxs[0]], population[idxs[1]]
                    X_new = X_i + np.random.rand(self.dim) * (X_r1 - X_r2)
                else:
                    # Regular Movement
                    idxs = np.random.choice(current_pop_size, 1, replace=False)
                    X_r = population[idxs[0]]
                    X_new = X_i + np.random.rand(self.dim) * (X_r - X_i)
                
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
            
            fitness_history.append(best_fitness)
        
        return best_solution, best_fitness, fitness_history

