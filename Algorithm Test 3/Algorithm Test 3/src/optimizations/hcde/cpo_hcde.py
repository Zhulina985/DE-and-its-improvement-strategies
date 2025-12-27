"""
Crested Porcupine Optimizer with HCDE Strategies
Integrates HCDE strategies (entropy-based diversity, hybrid perturbation,
multi-level archives) into CPO algorithm.
"""

import numpy as np
from src.algorithms.cpo_algorithm import CrestedPorcupineOptimizer
from .hcde_adapter import HCDEAdapter


class CPO_HCDE(CrestedPorcupineOptimizer):
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        super().__init__(func, bounds, pop_size, max_generations)
        
        # Initialize HCDE adapter
        self.hcde = HCDEAdapter(self.dim, self.pop_size, self.max_generations, self.bounds)
    
    def optimize(self):
        # 1. Initialization
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        population = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        self.hcde.increment_nfes(self.pop_size)
        
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
            
            # 2.0 Cyclical Population Reduction
            step = int(t / (self.max_generations / self.T))
            if step >= self.T: 
                step = self.T - 1
            
            target_size = int(max(self.N_min, self.pop_size - step * ((self.pop_size - self.N_min) / self.T)))
            
            if target_size < current_pop_size:
                population = population[:target_size]
                fitness = fitness[:target_size]
                current_pop_size = target_size
                self.hcde.pop_size = current_pop_size
                self.hcde.count = self.hcde.count[:current_pop_size]
            
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
            
            # Get adaptive parameter from HCDE
            adaptive_param = self.hcde.get_adaptive_parameter()
            
            new_population = np.zeros_like(population)
            
            for i in range(current_pop_size):
                X_i = population[i]
                
                # Random candidates
                candidates = [idx for idx in range(current_pop_size) if idx != i]
                if len(candidates) < 3:
                    r_indices = [np.random.randint(0, current_pop_size) for _ in range(3)]
                else:
                    r_indices = np.random.choice(candidates, 3, replace=False)
                r1, r2, r3 = r_indices[0], r_indices[1], r_indices[2]
                
                # Try to use archive individuals
                X_r1 = population[r1]
                if len(self.hcde.archive_A) > 0 and np.random.rand() < 0.3:
                    archive_ind = self.hcde.select_from_archive('A')
                    if archive_ind is not None:
                        X_r1 = archive_ind
                
                tau6 = np.random.rand()
                tau7 = np.random.rand()
                tau10 = np.random.rand()
                
                # Use adaptive parameter to influence exploration/exploitation
                exploration_prob = 0.5 * adaptive_param
                
                if np.random.rand() < exploration_prob:
                    # === EXPLORATION PHASE ===
                    
                    if tau6 < tau7:
                        r = r1 
                        y_i = (X_i + X_r1) / 2.0
                        I1 = np.random.normal()
                        tau2 = np.random.rand()
                        new_pos = X_CP + I1 * (y_i - tau2 * X_i)
                    else:
                        r = r1
                        y_i = (X_i + X_r1) / 2.0
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        diff_sign = np.sign(population[r1] - population[r2])
                        new_pos = X_i + U1 * (y_i - diff_sign * X_i)
                
                else:
                    # === EXPLOITATION PHASE ===
                    
                    S_i = np.exp(fitness[i] / sum_fitness)
                    delta = 1.0 if np.random.rand() > 0.5 else -1.0
                    
                    if tau10 < self.Tf:
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        tau3 = np.random.rand()
                        term2 = X_r1 + S_i * (population[r2] - population[r3]) - \
                                tau3 * delta * gamma_t * S_i
                        new_pos = (1 - U1) * X_i + U1 * term2
                    else:
                        tau4 = np.random.rand()
                        tau5 = np.random.rand()
                        v_next = X_r1
                        v_curr = X_i
                        m_i = np.exp(fitness[i] / sum_fitness)
                        F_i = tau6 * m_i * (v_next - v_curr)
                        factor = self.alpha * (1 - tau4) + tau4
                        term_mid = delta * X_CP - X_i
                        term_last = tau5 * delta * gamma_t * F_i
                        new_pos = X_CP + factor * term_mid - term_last
                
                # Boundary clamping
                new_pos = np.clip(new_pos, lb, ub)
                new_population[i] = new_pos
            
            # Evaluate New Population
            for i in range(current_pop_size):
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
            for i in range(current_pop_size):
                if self.hcde.count[i] == 0 and i < len(population):
                    f_updated = self.func(population[i])
                    self.hcde.increment_nfes(1)
                    fitness[i] = f_updated
                    if f_updated < best_fitness:
                        best_fitness = f_updated
                        best_solution = population[i].copy()
            
            # Sort population for next iteration
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

