"""
Crested Porcupine Optimizer with mLSHADE-SPACMA Strategies
Integrates mLSHADE-SPACMA strategies:
1. Precise elimination and generation mechanism
2. Modified mutation strategy with semi-parameter adaptation and RSP
3. Elite archiving mechanism
"""

import numpy as np
from src.algorithms.cpo_algorithm import CrestedPorcupineOptimizer
from .mlshade_spacma_adapter import MLSHADESPACMAAdapter


class CPO_MLSHADESPACMA(CrestedPorcupineOptimizer):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, rho=0.1):
        super().__init__(func, bounds, pop_size, max_generations)
        
        # Initialize mLSHADE-SPACMA adapter
        self.adapter = MLSHADESPACMAAdapter(
            self.func, self.dim, self.pop_size, self.max_generations, self.bounds, rho
        )
        self.adapter.increment_nfes(self.pop_size)
    
    def optimize(self):
        # 1. Initialization
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        population = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        # Sort to find best
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        best_solution = population[0].copy()
        best_fitness = fitness[0]
        
        fitness_history = []
        current_pop_size = self.pop_size
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # Apply precise elimination and generation (first half only)
            population, fitness = self.adapter.precise_elimination_and_generation(
                population, fitness
            )
            current_pop_size = len(population)
            
            # 2.0 Cyclical Population Reduction
            step = int(t / (self.max_generations / self.T))
            if step >= self.T: 
                step = self.T - 1
            
            target_size = int(max(self.N_min, self.pop_size - step * ((self.pop_size - self.N_min) / self.T)))
            
            if target_size < current_pop_size:
                population = population[:target_size]
                fitness = fitness[:target_size]
                current_pop_size = target_size
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx].copy()
                
            X_CP = best_solution
            
            # Pre-calculate common factors
            ratio = t / self.max_generations
            if ratio >= 1.0: ratio = 0.9999
            gamma_t = 2 * np.random.rand() * ((1 - ratio) ** ratio)
            sum_fitness = np.sum(np.abs(fitness)) + 1e-10
            
            new_population = np.zeros_like(population)
            successful_F = []
            
            # Select pbest (top p% of population)
            p = 0.11  # Typical p value
            p_size = max(1, int(p * current_pop_size))
            pbest_indices = np.random.choice(
                min(p_size, current_pop_size), 
                size=current_pop_size, 
                replace=True
            )
            
            for i in range(current_pop_size):
                X_i = population[i]
                X_pbest = population[pbest_indices[i]]
                
                # Get adaptive F
                r_i = np.random.randint(1, self.adapter.H + 1)
                F_i = self.adapter.get_adaptive_F(r_i)
                
                # Apply mLSHADE-SPACMA mutation strategy
                v_i = self.adapter.apply_mutation_strategy(
                    X_i, X_pbest, population, fitness, F_i
                )
                
                # CPO-specific update (simplified, can be enhanced)
                # Use v_i as base, then apply CPO movement
                candidates = [idx for idx in range(current_pop_size) if idx != i]
                if len(candidates) < 3:
                    r_indices = [np.random.randint(0, current_pop_size) for _ in range(3)]
                else:
                    r_indices = np.random.choice(candidates, size=3, replace=False)
                
                X_r1, X_r2, X_r3 = population[r_indices[0]], population[r_indices[1]], population[r_indices[2]]
                
                # CPO movement with mLSHADE-SPACMA guidance
                if np.random.rand() < self.Tf:
                    # Odor-based movement
                    S_i = (fitness[i] / sum_fitness) * gamma_t * (X_r1 - X_i)
                    X_new = X_i + S_i
                else:
                    # Physical movement
                    S_i = (fitness[i] / sum_fitness) * gamma_t * (X_r2 - X_r3)
                    X_new = X_i + S_i
                
                # Blend with mLSHADE-SPACMA mutation
                alpha = 0.5  # Blending factor
                X_new = alpha * v_i + (1 - alpha) * X_new
                
                # Apply bounds
                X_new = np.clip(X_new, lb, ub)
                
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

