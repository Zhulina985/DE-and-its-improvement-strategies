import numpy as np

class FloodAlgorithm:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, Ne=5):
        """
        Flood Algorithm (Formerly Fallow Land Algorithm) Implementation.
        
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
            
            # --- Update Step 1: Regular Movement vs Flood Disturbance ---
            
            f_min = np.min(fitness)
            f_max = np.max(fitness)
            denom = f_max - f_min
            if denom == 0: denom = 1e-10 # Avoid divide by zero
            
            # Calculate Pk (Water Loss Coefficient) for Flood Disturbance
            # Formula: Pk = (1.2 / t) * [ (sqrt(T*t^2 + 1) * ln(sqrt(T*t^2+1) + T/4)) / ( (T/4)*t ) ]^(-2/3)
            # This complex formula seems to serve as a decay factor.
            # Simplified interpretation based on visual structure:
            T = self.max_generations
            
            # Inner term A: sqrt(T * t^2 + 1)
            term_A = np.sqrt(T * (t**2) + 1)
            
            # Inner term B: ln(term_A + T/4)
            term_B = np.log(term_A + T/4.0)
            
            # Denominator inner: (T/4) * t
            term_denom = (T/4.0) * t
            
            # Bracket content
            bracket = (term_A * term_B) / term_denom
            
            # Pk
            if bracket <= 0: bracket = 1e-10
            Pk = (1.2 / t) * (bracket ** (-2.0/3.0))
            
            new_population = np.zeros_like(population)
            
            for i in range(self.pop_size):
                # Calculate Pe_i (Soil Permeability)
                Pe_i = ((fitness[i] - f_min) / denom) ** 2
                
                # Decision: Flood Disturbance if rand > Pe_i (Small Pe -> High Prob)
                # Note: User image had "rand > rand + Pe", we corrected to "rand > Pe" based on logic "Small Pe -> High Flood Prob"
                # If Pe=0 (Best), rand > 0 is true 100% -> Flood (Exploit). Correct.
                # If Pe=1 (Worst), rand > 1 is false -> Regular (Explore). Correct.
                
                if np.random.rand() > Pe_i:
                    # === Flood Disturbance (Local Exploitation) ===
                    # S_new = S_i + (Pk)^randn / t * (rand * (Smax - Smin) + Smin)
                    
                    # randn is standard normal
                    randn = np.random.randn()
                    
                    # Random position in space
                    rand_pos = self.lb + np.random.rand(self.dim) * self.range_width
                    
                    # Since Pk can be large or small, and raised to power randn (can be negative), 
                    # we must be careful with base being negative? Pk should be > 0.
                    # Pk formula has squares and sqrts, likely positive.
                    
                    # Factor calculation
                    try:
                        factor = (Pk ** randn) / t
                    except:
                        factor = 0.0 # Fallback
                        
                    step = factor * rand_pos
                    new_pos = population[i] + step
                    
                else:
                    # === Regular Movement (Global Exploration) ===
                    # S_new = S_best + rand * (S_j - S_i)
                    
                    # Select random S_j
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
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            # --- Update Step 2: Population Update (Water Increase/Decrease) ---
            # Trigger probability Pt = |sin(pi * rand)|
            Pt = np.abs(np.sin(np.pi * np.random.rand()))
            
            if np.random.rand() < Pt:
                # Triggered: Remove Ne worst, Add Ne new
                
                # Sort current population
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices]
                fitness = fitness[sorted_indices]
                
                # Note: sorted_indices[0] is best, [-1] is worst.
                # We replace the worst Ne individuals (indices from pop_size - Ne to pop_size - 1)
                
                start_bad_idx = self.pop_size - self.Ne
                
                for k in range(start_bad_idx, self.pop_size):
                    # Generate new individual
                    # Formula: S_new = S_best + rand * (rand * (Smax - Smin) + Smin)
                    
                    # Inner: random point in bounds
                    rand_point = self.lb + np.random.rand(self.dim) * self.range_width
                    
                    # New pos
                    new_ind = best_solution + np.random.rand() * rand_point
                    
                    # Clip
                    new_ind = np.clip(new_ind, self.lb, self.ub)
                    
                    # Update (Replace worst)
                    population[k] = new_ind
                    fitness[k] = self.func(new_ind)
                    
                    # Check if new random one is somehow best (unlikely but possible)
                    if fitness[k] < best_fitness:
                        best_fitness = fitness[k]
                        best_solution = population[k].copy()
            
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

