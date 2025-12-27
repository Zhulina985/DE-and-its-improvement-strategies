"""
Flood Algorithm with HCDE Strategies
Integrates HCDE strategies into Flood Algorithm.
"""

import numpy as np
from src.algorithms.flood_algorithm import FloodAlgorithm
from .hcde_adapter import HCDEAdapter


class FLOOD_HCDE(FloodAlgorithm):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, Ne=5):
        super().__init__(func, bounds, pop_size, max_generations, Ne)
        
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
            
            # --- Update Step 1: Regular Movement vs Flood Disturbance ---
            
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
            
            # Use adaptive parameter to influence flood disturbance probability
            adaptive_param = self.hcde.get_adaptive_parameter()
            
            new_population = np.zeros_like(population)
            
            for i in range(self.pop_size):
                # Calculate Pe_i (Soil Permeability)
                Pe_i = ((fitness[i] - f_min) / denom) ** 2
                
                # Adjust probability with adaptive parameter
                flood_prob = (1.0 - Pe_i) * adaptive_param
                
                if np.random.rand() < flood_prob:
                    # === Flood Disturbance (Local Exploitation) ===
                    randn = np.random.randn()
                    rand_pos = self.lb + np.random.rand(self.dim) * self.range_width
                    
                    # Try to use archive A for exploitation
                    if len(self.hcde.archive_A) > 0 and np.random.rand() < 0.3:
                        archive_ind = self.hcde.select_from_archive('A')
                        if archive_ind is not None:
                            rand_pos = archive_ind
                    
                    try:
                        factor = (Pk ** randn) / t
                    except:
                        factor = 0.0
                        
                    step = factor * rand_pos
                    new_pos = population[i] + step
                    
                else:
                    # === Regular Movement (Global Exploration) ===
                    j = np.random.randint(0, self.pop_size)
                    S_j = population[j]
                    
                    # Try to use archive B for exploration
                    if len(self.hcde.archive_B) > 0 and np.random.rand() < 0.3:
                        archive_ind = self.hcde.select_from_archive('B')
                        if archive_ind is not None:
                            S_j = archive_ind
                    
                    new_pos = best_solution + np.random.rand() * (S_j - population[i])
                
                # Boundary Check
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_population[i] = new_pos
            
            # Greedy Selection for Step 1
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                self.hcde.increment_nfes(1)
                
                if f_new < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            # --- Update Step 2: Population Update (Water Increase/Decrease) ---
            Pt = np.abs(np.sin(np.pi * np.random.rand()))
            
            if np.random.rand() < Pt:
                # Triggered: Remove Ne worst, Add Ne new
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices]
                fitness = fitness[sorted_indices]
                
                start_bad_idx = self.pop_size - self.Ne
                
                for k in range(start_bad_idx, self.pop_size):
                    rand_point = self.lb + np.random.rand(self.dim) * self.range_width
                    
                    # Try to use archive for new individuals
                    if len(self.hcde.archive_A) > 0 and np.random.rand() < 0.5:
                        archive_ind = self.hcde.select_from_archive('A')
                        if archive_ind is not None:
                            rand_point = archive_ind
                    
                    new_ind = best_solution + np.random.rand() * rand_point
                    new_ind = np.clip(new_ind, self.lb, self.ub)
                    
                    population[k] = new_ind
                    fitness[k] = self.func(new_ind)
                    self.hcde.increment_nfes(1)
                    
                    if fitness[k] < best_fitness:
                        best_fitness = fitness[k]
                        best_solution = population[k].copy()
            
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

