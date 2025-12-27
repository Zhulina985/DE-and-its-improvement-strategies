import numpy as np
from .dg_core import InsightsGuider

class FGO_DG:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, M=0.6, Ep=0.7):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.M = M
        self.Ep = Ep
        
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
            
            E_r = self.M + (1.0 - self.M) * (1.0 - t / self.max_generations)
            
            epsilon = 1e-10
            quality = (f_max - fitness) / (denom + epsilon)
            sum_quality = np.sum(quality) + epsilon
            v_vec = np.exp(quality / sum_quality)
            
            new_population = np.zeros_like(population)
            
            for i in range(self.pop_size):
                p_i = (fitness[i] - f_min) / (denom + epsilon)
                S_i = population[i]
                
                # FGO Logic
                if p_i < E_r:
                    r9 = np.random.rand()
                    r10 = np.random.rand()
                    if r9 < r10:
                        r1 = np.random.rand()
                        decay = 3.0 * (1.0 - t / self.max_generations)
                        v_adj = v_vec[i] * r1 * decay
                        idxs = np.random.choice(self.pop_size, 2, replace=False)
                        S_a, S_c = population[idxs[0]], population[idxs[1]]
                        D = (S_a - S_c) * v_adj
                        r3 = np.random.rand()
                        mask = (np.random.rand(self.dim) < r3).astype(float)
                        S_new = S_i + D * mask
                    else:
                        r7 = np.random.rand()
                        if r7 < 0.5:
                            idxs = np.random.choice(self.pop_size, 3, replace=False)
                            S_a, S_b, S_c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
                            D_b1 = S_a - S_b
                            D_b2 = best_solution - S_c
                            r5 = np.random.rand()
                            D_branch = D_b1 if r5 > 0.5 else D_b2
                            r6, r3 = np.random.rand(), np.random.rand()
                            mask = (np.random.rand(self.dim) < r3).astype(float)
                            S_new = S_i + r6 * D_branch * mask
                        else:
                            idxs = np.random.choice(self.pop_size, 2, replace=False)
                            S_a, S_b = population[idxs[0]], population[idxs[1]]
                            S_spore = (best_solution + S_a + S_b) / 3.0
                            r1 = np.random.rand()
                            decay = 3.0 * (1.0 - t / self.max_generations)
                            v_adj = v_vec[i] * r1 * decay
                            S_g = 1.0 if np.random.rand() < 0.5 else -1.0
                            S_new = S_spore + S_g * v_adj * np.abs(S_spore - S_i)
                else:
                    if t < self.max_generations / 2:
                        ni = np.random.rand()
                    else:
                        ni = quality[i] / sum_quality
                    
                    idxs = np.random.choice(self.pop_size, 1, replace=False)
                    S_a = population[idxs[0]]
                    beta = 1.0 if np.random.rand() < 0.5 else -1.0
                    r5 = np.random.rand()
                    D_chem1 = r5 * (beta * best_solution + (1.0 - beta) * S_a - S_i)
                    
                    r9, r11, r13 = np.random.rand(), np.random.rand(), np.random.rand()
                    walk_trigger = 1.0 if r11 > r13 else 0.0
                    S_new = S_i + ni * D_chem1 + r9 * self.Ep * walk_trigger * (np.random.rand(self.dim) * 2 - 1)
                
                # NNOP Hook
                if self.guider.is_ready and np.random.rand() < self.tau:
                    nn_pos = self.guider.predict(S_new)
                    nn_pos = np.clip(nn_pos, self.lb, self.ub)
                    f_nn = self.func(nn_pos)
                    f_std = self.func(np.clip(S_new, self.lb, self.ub))
                    
                    if f_nn < f_std:
                        S_new = nn_pos
                        f_new = f_nn
                    else:
                        f_new = f_std
                else:
                    S_new = np.clip(S_new, self.lb, self.ub)
                    f_new = self.func(S_new)
                
                new_population[i] = S_new
                
                # Selection
                if f_new < fitness[i]:
                    self.guider.store(population[i], new_population[i])
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

