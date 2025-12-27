import numpy as np
from .tsms_utils import TSMS_Utils, Archive

class FGO_TSMS:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, M=0.6, Ep=0.7):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.M = M
        self.Ep = Ep
        
        self.archive_B = Archive(capacity=4*pop_size)
        self.archive_A = Archive(capacity=pop_size)

    def optimize(self):
        population, fitness = TSMS_Utils.obl_initialization(self.func, self.bounds, self.pop_size, self.dim)
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        rho = int(0.66 * self.max_generations)
        
        for t in range(1, self.max_generations + 1):
            if t < rho:
                pool = self.archive_B.get_candidates(population)
            else:
                pool = self.archive_A.get_candidates(population)
                
            f_min, f_max = np.min(fitness), np.max(fitness)
            denom = f_max - f_min if (f_max - f_min) != 0 else 1e-10
            E_r = self.M + (1.0 - self.M) * (1.0 - t / self.max_generations)
            
            epsilon = 1e-10
            quality = (f_max - fitness) / (denom + epsilon)
            sum_quality = np.sum(quality) + epsilon
            v_vec = np.exp(quality / sum_quality)
            
            new_population = np.zeros_like(population)
            failed_trials = []
            
            for i in range(self.pop_size):
                # Pick Archive Candidates
                if len(pool) > 0:
                    X_arc1 = pool[np.random.randint(0, len(pool))]
                    X_arc2 = pool[np.random.randint(0, len(pool))]
                else:
                    X_arc1 = population[np.random.randint(0, self.pop_size)]
                    X_arc2 = population[np.random.randint(0, self.pop_size)]
                
                p_i = (fitness[i] - f_min) / (denom + epsilon)
                S_i = population[i]
                
                if p_i < E_r:
                    if np.random.rand() < np.random.rand():
                        # Tip Growth: Use Archive for S_c
                        S_a = population[np.random.randint(0, self.pop_size)]
                        S_c = X_arc1 # Archive sub
                        
                        r1 = np.random.rand()
                        decay = 3.0 * (1.0 - t / self.max_generations)
                        v_adj = v_vec[i] * r1 * decay
                        D = (S_a - S_c) * v_adj
                        mask = (np.random.rand(self.dim) < np.random.rand()).astype(float)
                        S_new = S_i + D * mask
                    else:
                        if np.random.rand() < 0.5:
                            # Branching: Use Archive
                            S_a = population[np.random.randint(0, self.pop_size)]
                            S_b = X_arc1
                            S_c = X_arc2
                            D_b1 = S_a - S_b
                            D_b2 = best_solution - S_c
                            D_branch = D_b1 if np.random.rand() > 0.5 else D_b2
                            S_new = S_i + np.random.rand() * D_branch * (np.random.rand(self.dim) < np.random.rand())
                        else:
                            # Spore
                            S_a = population[np.random.randint(0, self.pop_size)]
                            S_b = X_arc1
                            S_spore = (best_solution + S_a + S_b) / 3.0
                            v_adj = v_vec[i] * np.random.rand() * 3.0 * (1 - t/self.max_generations)
                            S_g = 1.0 if np.random.rand() < 0.5 else -1.0
                            S_new = S_spore + S_g * v_adj * np.abs(S_spore - S_i)
                else:
                    # Exploitation: Archive for S_a?
                    S_a = X_arc1
                    if t < self.max_generations / 2: ni = np.random.rand()
                    else: ni = quality[i] / sum_quality
                    
                    beta = 1.0 if np.random.rand() < 0.5 else -1.0
                    D_chem1 = np.random.rand() * (beta * best_solution + (1.0 - beta) * S_a - S_i)
                    walk = 1.0 if np.random.rand() > np.random.rand() else 0.0
                    S_new = S_i + ni * D_chem1 + np.random.rand() * self.Ep * walk * (np.random.rand(self.dim) * 2 - 1)
                
                S_new = np.clip(S_new, lb, ub)
                new_population[i] = S_new
            
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                if f_new < fitness[i]:
                    self.archive_B.add([population[i]])
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                else:
                    failed_trials.append(new_population[i])
            
            if failed_trials:
                self.archive_A.add(failed_trials)
                
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

