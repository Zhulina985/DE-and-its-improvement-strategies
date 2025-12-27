import numpy as np
from .dg_core import InsightsGuider

class FLOOD_DG:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, Ne=5):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.Ne = Ne
        
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.range_width = self.ub - self.lb
        
        self.guider = InsightsGuider(self.dim)
        self.tau = 0.1

    def optimize(self):
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        
        for t in range(1, self.max_generations + 1):
            
            if len(self.guider.data_X) > 32:
                self.guider.train()
            
            f_min = np.min(fitness)
            f_max = np.max(fitness)
            denom = f_max - f_min
            if denom == 0: denom = 1e-10
            
            T = self.max_generations
            term_A = np.sqrt(T * (t**2) + 1)
            term_B = np.log(term_A + T/4.0)
            term_denom = (T/4.0) * t
            bracket = (term_A * term_B) / term_denom
            if bracket <= 0: bracket = 1e-10
            Pk = (1.2 / t) * (bracket ** (-2.0/3.0))
            
            new_population = np.zeros_like(population)
            
            # Step 1
            for i in range(self.pop_size):
                Pe_i = ((fitness[i] - f_min) / denom) ** 2
                
                if np.random.rand() > Pe_i:
                    randn = np.random.randn()
                    rand_pos = self.lb + np.random.rand(self.dim) * self.range_width
                    try:
                        factor = (Pk ** randn) / t
                    except:
                        factor = 0.0
                    step = factor * rand_pos
                    new_pos = population[i] + step
                else:
                    j = np.random.randint(0, self.pop_size)
                    S_j = population[j]
                    new_pos = best_solution + np.random.rand() * (S_j - population[i])
                
                # NNOP Hook
                if self.guider.is_ready and np.random.rand() < self.tau:
                    nn_pos = self.guider.predict(new_pos)
                    nn_pos = np.clip(nn_pos, self.lb, self.ub)
                    f_nn = self.func(nn_pos)
                    f_std = self.func(np.clip(new_pos, self.lb, self.ub))
                    
                    if f_nn < f_std:
                        new_pos = nn_pos
                        f_new = f_nn
                    else:
                        f_new = f_std
                else:
                    new_pos = np.clip(new_pos, self.lb, self.ub)
                    f_new = self.func(new_pos)
                
                new_population[i] = new_pos
                
                # Selection
                if f_new < fitness[i]:
                    self.guider.store(population[i], new_population[i])
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            # Step 2
            Pt = np.abs(np.sin(np.pi * np.random.rand()))
            if np.random.rand() < Pt:
                sorted_indices = np.argsort(fitness)
                population = population[sorted_indices]
                fitness = fitness[sorted_indices]
                
                start_bad_idx = self.pop_size - self.Ne
                for k in range(start_bad_idx, self.pop_size):
                    rand_point = self.lb + np.random.rand(self.dim) * self.range_width
                    new_ind = best_solution + np.random.rand() * rand_point
                    new_ind = np.clip(new_ind, self.lb, self.ub)
                    
                    f_val = self.func(new_ind)
                    
                    # Should we apply NNOP here too? Why not.
                    if self.guider.is_ready and np.random.rand() < self.tau:
                        nn_ind = self.guider.predict(new_ind)
                        nn_ind = np.clip(nn_ind, self.lb, self.ub)
                        f_nn = self.func(nn_ind)
                        if f_nn < f_val:
                            new_ind = nn_ind
                            f_val = f_nn
                            
                    population[k] = new_ind
                    fitness[k] = f_val
                    
                    if fitness[k] < best_fitness:
                        best_fitness = fitness[k]
                        best_solution = population[k].copy()
                        
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

