"""
Wave Optics Optimizer with HCDE Strategies
Integrates HCDE strategies into WOO algorithm.
"""

import numpy as np
from src.algorithms.woo_algorithm import WaveOpticsOptimizer
from .hcde_adapter import HCDEAdapter


class WOO_HCDE(WaveOpticsOptimizer):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        super().__init__(func, bounds, pop_size, max_generations)
        
        # Initialize HCDE adapter
        self.hcde = HCDEAdapter(self.dim, self.pop_size, self.max_generations, self.bounds)
    
    def optimize(self):
        # 1. Initialization
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        self.hcde.increment_nfes(self.pop_size)
        
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
            
            # Dynamic Parameters
            C = 1.0 - t / self.max_generations
            pw = np.sin((1.0 - t/self.max_generations) * np.pi / 2.0)
            
            tau1 = 0.2 * C + 0.8
            tau2_thresh = np.exp(-tau1)
            
            # Use adaptive parameter to influence strategy selection
            adaptive_param = self.hcde.get_adaptive_parameter()
            tau2_thresh = tau2_thresh * adaptive_param
            
            # Calculate Mid index for U1 (Interference)
            T_val = int(abs(round((1.0 - np.sin(np.pi/2.0 * self.Ua)) * self.pop_size)))
            T_val = max(0, min(self.pop_size-1, T_val))
            
            new_population = np.zeros_like(population)
            
            for i in range(self.pop_size):
                X_i = population[i]
                
                # --- Strategy Selection ---
                if np.random.rand() < tau2_thresh:
                    # === Random Strategy (Exploration) ===
                    
                    Rd1 = np.random.randn(self.dim)
                    R1 = np.random.rand()
                    
                    term1 = (R1 * np.abs(Rd1)) ** (C + 3.0)
                    om = term1 + 0.01 * np.random.randn(self.dim)
                    
                    r_rnd = np.random.rand()
                    s_val = np.exp(-2 * r_rnd * (1 - r_rnd))
                    if np.random.rand() < s_val:
                        Z1 = np.ones(self.dim)
                    else:
                        Z1 = np.random.rand(self.dim)
                    
                    if T_val > 0:
                        mid_idx = np.random.randint(0, T_val)
                    else:
                        mid_idx = 0
                    X_mid = population[mid_idx]
                    
                    # Try to use archive B for exploration
                    if len(self.hcde.archive_B) > 0 and np.random.rand() < 0.3:
                        archive_ind = self.hcde.select_from_archive('B')
                        if archive_ind is not None:
                            X_mid = archive_ind
                    
                    U1 = (X_mid - X_i) * pw
                    
                    r_indices = np.random.choice(self.pop_size, 3, replace=False)
                    if i < self.pop_size / 2 and np.random.rand() > C:
                        top_half = max(1, int(self.pop_size/2))
                        idx1 = np.random.randint(0, top_half)
                        idx2 = np.random.randint(0, top_half)
                        U2 = population[idx1] - population[idx2]
                    else:
                        U2 = population[r_indices[0]] - population[r_indices[1]]
                    
                    w = np.random.rand()
                    
                    if t < self.max_generations / 3:
                        DU1 = w * U1 + (1 - w) * U2
                    else:
                        SU = np.random.rand()
                        DU1 = U1 + SU * U2
                        
                    X_new = X_i + om * Z1 * DU1
                    
                else:
                    # === Intensification Strategy (Exploitation) ===
                    
                    if np.random.rand() > C:
                        Rd1 = np.random.randn(self.dim)
                        V = Rd1 * (np.sin(np.random.rand(self.dim) * 2 * np.pi))
                        
                        # Try to use archive A for exploitation
                        if len(self.hcde.archive_A) > 0 and np.random.rand() < 0.3:
                            archive_ind = self.hcde.select_from_archive('A')
                            if archive_ind is not None:
                                X_new = V * archive_ind
                            else:
                                X_new = V * best_solution
                        else:
                            X_new = V * best_solution
                    else:
                        r3 = np.random.rand()
                        
                        r_indices = np.random.choice(self.pop_size, 3, replace=False)
                        U2 = population[r_indices[0]] - population[r_indices[1]]
                        
                        Z2 = np.random.rand(self.dim)
                        om3 = np.random.rand(self.dim) * (1 - r3)
                        
                        Xr3 = population[r_indices[1]]
                        Xr4 = population[r_indices[2]]
                        
                        DX = Z2 * om3 * ( (best_solution + Xr3)/2.0 - Xr4 )
                        
                        X_new = X_i + r3 * np.exp(-1.0) * U2 + DX
                
                # Boundary Check
                X_new = np.clip(X_new, self.lb, self.ub)
                new_population[i] = X_new
                
            # Evaluation and Selection
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                self.hcde.increment_nfes(1)
                
                if f_new < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            # Sort for next iteration
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
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
            
            # Re-sort after diversity update
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            fitness_history.append(best_fitness)
            
            # 3. Population Adjustment (Uask update)
            self.Ua = self.Ua * 0.99
            
        return best_solution, best_fitness, fitness_history

