import numpy as np
from .dg_core import InsightsGuider

class WOO_DG:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.range_width = self.ub - self.lb
        
        self.lam = 5e-4 
        self.d = 1e-2   
        self.Ua = 0.5   
        
        self.guider = InsightsGuider(self.dim)
        self.tau = 0.1

    def optimize(self):
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        best_solution = population[0].copy()
        best_fitness = fitness[0]
        fitness_history = []
        
        for t in range(1, self.max_generations + 1):
            
            if len(self.guider.data_X) > 32:
                self.guider.train()
            
            C = 1.0 - t / self.max_generations
            pw = np.sin((1.0 - t/self.max_generations) * np.pi / 2.0)
            tau1 = 0.2 * C + 0.8
            tau2_thresh = np.exp(-tau1)
            
            new_population = np.zeros_like(population)
            
            T_val = int(abs(round((1.0 - np.sin(np.pi/2.0 * self.Ua)) * self.pop_size)))
            T_val = max(0, min(self.pop_size-1, T_val))
            
            for i in range(self.pop_size):
                X_i = population[i]
                
                if np.random.rand() < tau2_thresh:
                    Rd1 = np.random.randn(self.dim)
                    R1 = np.random.rand()
                    term1 = (R1 * np.abs(Rd1)) ** (C + 3.0)
                    om = term1 + 0.01 * np.random.randn(self.dim)
                    
                    r_rnd = np.random.rand()
                    s_val = np.exp(-2 * r_rnd * (1 - r_rnd))
                    Z1 = np.ones(self.dim) if np.random.rand() < s_val else np.random.rand(self.dim)
                    
                    if T_val > 0: mid_idx = np.random.randint(0, T_val)
                    else: mid_idx = 0
                    X_mid = population[mid_idx]
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
                    if np.random.rand() > C:
                        Rd1 = np.random.randn(self.dim)
                        V = Rd1 * (np.sin(np.random.rand(self.dim) * 2 * np.pi))
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
                
                # NNOP Hook
                if self.guider.is_ready and np.random.rand() < self.tau:
                    nn_pos = self.guider.predict(X_new)
                    nn_pos = np.clip(nn_pos, self.lb, self.ub)
                    f_nn = self.func(nn_pos)
                    f_std = self.func(np.clip(X_new, self.lb, self.ub))
                    
                    if f_nn < f_std:
                        X_new = nn_pos
                        f_new = f_nn
                    else:
                        f_new = f_std
                else:
                    X_new = np.clip(X_new, self.lb, self.ub)
                    f_new = self.func(X_new)
                
                new_population[i] = X_new
                
                # Selection
                if f_new < fitness[i]:
                    self.guider.store(population[i], new_population[i])
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            fitness_history.append(best_fitness)
            self.Ua = self.Ua * 0.99
            
        return best_solution, best_fitness, fitness_history

