import numpy as np
from .tsms_utils import TSMS_Utils, Archive

class FLOOD_TSMS:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, Ne=5):
        """
        Flood Algorithm with TSMS Optimization Strategy.
        
        Args:
            func (callable): Objective function.
            bounds (list of tuple): [(min, max), ...].
            pop_size (int): Population size (N_pop).
            max_generations (int): Maximum iterations (Iter_max).
            Ne (int): Number of individuals to refresh in population update phase.
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.Ne = Ne
        
        # Bounds arrays
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.range_width = self.ub - self.lb
        
        # TSMS Components
        self.archive_B = Archive(capacity=4*pop_size)  # History archive
        self.archive_A = Archive(capacity=pop_size)    # Inferior archive

    def optimize(self):
        # 1. OBL Initialization
        population, fitness = TSMS_Utils.obl_initialization(self.func, self.bounds, self.pop_size, self.dim)
        
        # Find global best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        
        # TSMS Stage threshold
        rho = int(0.66 * self.max_generations)
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # --- TSMS Stage Determination ---            
            if t < rho:
                # Stage 1: Use Archive B (History)
                pool = self.archive_B.get_candidates(population)
            else:
                # Stage 2: Use Archive A (Inferior)
                pool = self.archive_A.get_candidates(population)
            
            # --- Update Step 1: Regular Movement vs Flood Disturbance ---
            
            f_min = np.min(fitness)
            f_max = np.max(fitness)
            denom = f_max - f_min
            if denom == 0: denom = 1e-10  # Avoid divide by zero
            
            # Calculate Pk (Water Loss Coefficient) for Flood Disturbance
            T = self.max_generations
            term_A = np.sqrt(T * (t**2) + 1)
            term_B = np.log(term_A + T/4.0)
            term_denom = (T/4.0) * t
            bracket = (term_A * term_B) / term_denom
            if bracket <= 0: bracket = 1e-10
            Pk = (1.2 / t) * (bracket ** (-2.0/3.0))
            
            new_population = np.zeros_like(population)
            failed_trials = []
            
            for i in range(self.pop_size):
                # Calculate Pe_i (Soil Permeability)
                Pe_i = ((fitness[i] - f_min) / denom) ** 2
                
                if np.random.rand() > Pe_i:
                    # === Flood Disturbance (Local Exploitation) ===
                    randn = np.random.randn()
                    
                    # Select random point from pool instead of random position
                    if len(pool) > 0:
                        rand_pos = pool[np.random.randint(0, len(pool))]
                    else:
                        rand_pos = self.lb + np.random.rand(self.dim) * self.range_width
                    
                    try:
                        factor = (Pk ** randn) / t
                    except:
                        factor = 0.0
                        
                    step = factor * rand_pos
                    new_pos = population[i] + step
                    
                else:
                    # === Regular Movement (Global Exploration) ===
                    # Select random S_j from pool
                    if len(pool) > 0:
                        j = np.random.randint(0, len(pool))
                        S_j = pool[j]
                    else:
                        j = np.random.randint(0, self.pop_size)
                        S_j = population[j]
                    
                    new_pos = best_solution + np.random.rand() * (S_j - population[i])
                
                # Boundary Check
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_population[i] = new_pos
            
            # Greedy Selection for Step 1
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                if f_new < fitness[i]:
                    self.archive_B.add([population[i]])
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = population[i].copy()
                else:
                    failed_trials.append(new_population[i])
            
            # Update Inferior Archive
            if failed_trials:
                self.archive_A.add(failed_trials)
            
            # --- Update Step 2: Population Update (Water Increase/Decrease) ---
            # Trigger probability Pt = |sin(pi * rand)|
            Pt = np.abs(np.sin(np.pi * np.random.rand()))
            
            if np.random.rand() < Pt:
                # Triggered: Remove Ne worst, Add Ne new
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices]
                fitness = fitness[sorted_indices]
                
                start_bad_idx = self.pop_size - self.Ne
                
                for k in range(start_bad_idx, self.pop_size):
                    # Generate new individual
                    rand_point = self.lb + np.random.rand(self.dim) * self.range_width
                    new_ind = best_solution + np.random.rand() * rand_point
                    new_ind = np.clip(new_ind, self.lb, self.ub)
                    
                    population[k] = new_ind
                    fitness[k] = self.func(new_ind)
                    
                    if fitness[k] < best_fitness:
                        best_fitness = fitness[k]
                        best_solution = population[k].copy()
            
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history