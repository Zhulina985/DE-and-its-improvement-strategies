"""
Fungal Growth Optimizer with mLSHADE-SPACMA Strategies
Integrates mLSHADE-SPACMA strategies into FGO algorithm.
"""

import numpy as np
from src.algorithms.fgo_algorithm import FungalGrowthOptimizer
from .mlshade_spacma_adapter import MLSHADESPACMAAdapter


class FGO_MLSHADESPACMA(FungalGrowthOptimizer):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, M=0.6, Ep=0.7, rho=0.1):
        super().__init__(func, bounds, pop_size, max_generations, M, Ep)
        
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
            
            # Exploration Threshold E_r
            E_r = self.M + (1.0 - self.M) * (1.0 - t / self.max_generations)
            
            # Growth Rate Base v_i calculations
            epsilon = 1e-10
            quality = (f_max - fitness) / (denom + epsilon)
            sum_quality = np.sum(quality) + epsilon
            v_vec = np.exp(quality / sum_quality)
            
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
                S_i = population[i]
                
                # Get adaptive F
                r_i = np.random.randint(1, self.adapter.H + 1)
                F_i = self.adapter.get_adaptive_F(r_i)
                
                # Apply mLSHADE-SPACMA mutation strategy
                X_pbest = population[pbest_indices[i]]
                v_i = self.adapter.apply_mutation_strategy(
                    S_i, X_pbest, population, fitness, F_i
                )
                
                # FGO-specific update
                if np.random.rand() < E_r:
                    # Exploration: Spore Dispersal
                    idxs = np.random.choice(current_pop_size, 2, replace=False)
                    S_spore = population[idxs[0]]
                    S_g = population[idxs[1]]
                    v_adj = v_vec[i] / (np.sum(v_vec) + epsilon)
                    S_new = S_spore + S_g * v_adj * np.abs(S_spore - S_i)
                else:
                    # Exploitation: Chemotaxis
                    ni = np.random.rand() if t < self.max_generations / 2 else quality[i] / sum_quality
                    idxs = np.random.choice(current_pop_size, 1, replace=False)
                    S_a = population[idxs[0]]
                    beta = 1.0 if np.random.rand() < 0.5 else -1.0
                    r5 = np.random.rand()
                    D_chem1 = r5 * (beta * best_solution + (1.0 - beta) * S_a - S_i)
                    r9, r11, r13 = np.random.rand(), np.random.rand(), np.random.rand()
                    walk_trigger = 1.0 if r11 > r13 else 0.0
                    S_new = S_i + ni * D_chem1 + r9 * self.Ep * walk_trigger * (np.random.rand(self.dim) * 2 - 1)
                
                # Blend with mLSHADE-SPACMA mutation
                alpha = 0.5
                S_new = alpha * v_i + (1 - alpha) * S_new
                
                # Apply bounds
                S_new = np.clip(S_new, self.lb, self.ub)
                
                # Evaluate
                f_new = self.func(S_new)
                self.adapter.increment_nfes(1)
                
                # Selection
                if f_new < fitness[i]:
                    # Update archive
                    self.adapter.update_archive(population[i], S_new, fitness[i], f_new)
                    successful_F.append(F_i)
                    
                    population[i] = S_new
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = S_new.copy()
                
                new_population[i] = population[i]
            
            # Update memory
            self.adapter.update_memory_F(successful_F)
            
            fitness_history.append(best_fitness)
        
        return best_solution, best_fitness, fitness_history

