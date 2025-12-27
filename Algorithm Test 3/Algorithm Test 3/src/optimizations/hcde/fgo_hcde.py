"""
Fungal Growth Optimizer with HCDE Strategies
Integrates HCDE strategies into FGO algorithm.
"""

import numpy as np
from src.algorithms.fgo_algorithm import FungalGrowthOptimizer
from .hcde_adapter import HCDEAdapter


class FGO_HCDE(FungalGrowthOptimizer):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, M=0.6, Ep=0.7):
        super().__init__(func, bounds, pop_size, max_generations, M, Ep)
        
        # Initialize HCDE adapter
        self.hcde = HCDEAdapter(self.dim, self.pop_size, self.max_generations, self.bounds)
    
    def optimize(self):
        # 1. Initialization
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        self.hcde.increment_nfes(self.pop_size)
        
        # Find global best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # Update global stats
            f_min = np.min(fitness)
            f_max = np.max(fitness)
            denom = f_max - f_min
            if denom == 0: denom = 1e-10
            
            # Use adaptive parameter from HCDE to influence exploration threshold
            adaptive_param = self.hcde.get_adaptive_parameter()
            E_r = self.M + (1.0 - self.M) * (1.0 - t / self.max_generations) * adaptive_param
            
            # Growth Rate Base v_i calculations
            epsilon = 1e-10
            quality = (f_max - fitness) / (denom + epsilon)
            sum_quality = np.sum(quality) + epsilon
            
            v_vec = np.exp(quality / sum_quality)
            
            new_population = np.zeros_like(population)
            
            for i in range(self.pop_size):
                # Normalized fitness p_i for switching rule
                p_i = (fitness[i] - f_min) / (denom + epsilon)
                
                S_i = population[i]
                
                # Switch Rule
                if p_i < E_r:
                    # === Exploration Phase ===
                    
                    r9 = np.random.rand()
                    r10 = np.random.rand()
                    
                    # 1. Hyphal Tip Growth
                    if r9 < r10:
                        r1 = np.random.rand()
                        decay = 3.0 * (1.0 - t / self.max_generations)
                        v_adj = v_vec[i] * r1 * decay
                        
                        idxs = np.random.choice(self.pop_size, 2, replace=False)
                        S_a = population[idxs[0]]
                        S_c = population[idxs[1]]
                        
                        # Try to use archive
                        if len(self.hcde.archive_B) > 0 and np.random.rand() < 0.3:
                            archive_ind = self.hcde.select_from_archive('B')
                            if archive_ind is not None:
                                S_a = archive_ind
                        
                        D = (S_a - S_c) * v_adj
                        r3 = np.random.rand()
                        mask = (np.random.rand(self.dim) < r3).astype(float)
                        S_new = S_i + D * mask
                    else:
                        r7 = np.random.rand()
                        
                        # 2. Hyphal Branching
                        if r7 < 0.5:
                            idxs = np.random.choice(self.pop_size, 3, replace=False)
                            S_a = population[idxs[0]]
                            S_b = population[idxs[1]]
                            S_c = population[idxs[2]]
                            
                            D_b1 = S_a - S_b
                            D_b2 = best_solution - S_c
                            
                            r5 = np.random.rand()
                            if r5 > 0.5:
                                D_branch = D_b1
                            else:
                                D_branch = D_b2
                                
                            r6 = np.random.rand()
                            r3 = np.random.rand()
                            mask = (np.random.rand(self.dim) < r3).astype(float)
                            
                            S_new = S_i + r6 * D_branch * mask
                            
                        # 3. Spore Germination
                        else:
                            idxs = np.random.choice(self.pop_size, 2, replace=False)
                            S_a = population[idxs[0]]
                            S_b = population[idxs[1]]
                            
                            S_spore = (best_solution + S_a + S_b) / 3.0
                            
                            r1 = np.random.rand()
                            decay = 3.0 * (1.0 - t / self.max_generations)
                            v_adj = v_vec[i] * r1 * decay
                            
                            S_g = 1.0 if np.random.rand() < 0.5 else -1.0
                            
                            S_new = S_spore + S_g * v_adj * np.abs(S_spore - S_i)
                
                else:
                    # === Exploitation Phase (Chemotropism) ===
                    
                    if t < self.max_generations / 2:
                        ni = np.random.rand()
                    else:
                        ni = quality[i] / sum_quality
                    
                    idxs = np.random.choice(self.pop_size, 1, replace=False)
                    S_a = population[idxs[0]]
                    
                    # Try to use archive A
                    if len(self.hcde.archive_A) > 0 and np.random.rand() < 0.3:
                        archive_ind = self.hcde.select_from_archive('A')
                        if archive_ind is not None:
                            S_a = archive_ind
                    
                    beta = 1.0 if np.random.rand() < 0.5 else -1.0
                    
                    r5 = np.random.rand()
                    D_chem1 = r5 * (beta * best_solution + (1.0 - beta) * S_a - S_i)
                    
                    r9 = np.random.rand()
                    r11 = np.random.rand()
                    r13 = np.random.rand()
                    
                    walk_trigger = 1.0 if r11 > r13 else 0.0
                    
                    S_new = S_i + ni * D_chem1 + r9 * self.Ep * walk_trigger * (np.random.rand(self.dim) * 2 - 1)
                
                # Boundary Check
                S_new = np.clip(S_new, self.lb, self.ub)
                new_population[i] = S_new
            
            # Selection (Greedy)
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                self.hcde.increment_nfes(1)
                
                if f_new < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
            
            # Update archives
            self.hcde.update_archives(population, fitness, best_solution, best_fitness)
            
            # Apply HCDE diversity enhancement mechanism
            population, fitness = self.hcde.update_stagnant_individuals(
                population, fitness, best_solution, best_fitness
            )
            
            # Re-evaluate updated individuals
            for i in range(self.pop_size):
                if self.hcde.count[i] == 0 and i < len(population):
                    f_updated = self.func(population[i])
                    self.hcde.increment_nfes(1)
                    fitness[i] = f_updated
                    if f_updated < best_fitness:
                        best_fitness = f_updated
                        best_solution = population[i].copy()
            
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

