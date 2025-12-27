import numpy as np
from .dg_core import InsightsGuider

class CPO_DG:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        self.T = 2
        self.alpha = 0.2
        self.Tf = 0.8
        self.N_min = max(5, int(self.pop_size * 0.1))
        
        # DG Components
        self.guider = InsightsGuider(self.dim)
        self.tau = 0.1 # Frequency control
        
    def optimize(self):
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        population = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        best_solution = population[0].copy()
        best_fitness = fitness[0]
        fitness_history = []
        current_pop_size = self.pop_size
        
        for t in range(1, self.max_generations + 1):
            
            # --- DG Training ---
            # Train periodically (e.g., every generation or every 5)
            # Paper suggests self-evolution is continuous.
            if len(self.guider.data_X) > 32:
                self.guider.train()
            
            step = int(t / (self.max_generations / self.T))
            if step >= self.T: step = self.T - 1
            target_size = int(max(self.N_min, self.pop_size - step * ((self.pop_size - self.N_min) / self.T)))
            
            if target_size < current_pop_size:
                population = population[:target_size]
                fitness = fitness[:target_size]
                current_pop_size = target_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx].copy()
            X_CP = best_solution
            
            ratio = t / self.max_generations
            if ratio >= 1.0: ratio = 0.9999
            gamma_t = 2 * np.random.rand() * ((1 - ratio) ** ratio)
            sum_fitness = np.sum(np.abs(fitness)) + 1e-10
            
            new_population = np.zeros_like(population)
            
            for i in range(current_pop_size):
                X_i = population[i]
                
                # Standard CPO Logic
                candidates = [idx for idx in range(current_pop_size) if idx != i]
                if len(candidates) < 3: 
                    r_indices = [np.random.randint(0, current_pop_size) for _ in range(3)]
                else:
                    r_indices = np.random.choice(candidates, 3, replace=False)
                r1, r2, r3 = r_indices[0], r_indices[1], r_indices[2]
                
                tau6, tau7, tau10 = np.random.rand(), np.random.rand(), np.random.rand()
                
                if np.random.rand() < 0.5:
                    if tau6 < tau7:
                        r = r1 
                        y_i = (X_i + population[r]) / 2.0
                        I1 = np.random.normal()
                        tau2 = np.random.rand()
                        new_pos = X_CP + I1 * (y_i - tau2 * X_i)
                    else:
                        r = r1
                        y_i = (X_i + population[r]) / 2.0
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        diff_sign = np.sign(population[r1] - population[r2])
                        new_pos = X_i + U1 * (y_i - diff_sign * X_i)
                else:
                    S_i = np.exp(fitness[i] / sum_fitness)
                    delta = 1.0 if np.random.rand() > 0.5 else -1.0
                    
                    if tau10 < self.Tf:
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        tau3 = np.random.rand()
                        term2 = population[r1] + S_i * (population[r2] - population[r3]) - \
                                tau3 * delta * gamma_t * S_i
                        new_pos = (1 - U1) * X_i + U1 * term2
                    else:
                        tau4, tau5 = np.random.rand(), np.random.rand()
                        v_next, v_curr = population[r1], X_i
                        m_i = np.exp(fitness[i] / sum_fitness)
                        F_i = tau6 * m_i * (v_next - v_curr)
                        factor = self.alpha * (1 - tau4) + tau4
                        term_mid = delta * X_CP - X_i
                        term_last = tau5 * delta * gamma_t * F_i
                        new_pos = X_CP + factor * term_mid - term_last
                
                # --- NNOP Application ---
                # Apply with probability tau, or specific index rule
                # Simple probabilistic application
                if self.guider.is_ready and np.random.rand() < self.tau:
                    nn_pos = self.guider.predict(new_pos)
                    # Boundary check
                    nn_pos = np.clip(nn_pos, lb, ub)
                    # Evaluate NN solution
                    f_nn = self.func(nn_pos)
                    f_std = self.func(np.clip(new_pos, lb, ub)) # Eval standard
                    
                    if f_nn < f_std:
                        new_pos = nn_pos
                        f_new = f_nn
                    else:
                        f_new = f_std
                else:
                    new_pos = np.clip(new_pos, lb, ub)
                    f_new = self.func(new_pos)
                
                new_population[i] = new_pos
                
                # Selection & Collection
                if f_new < fitness[i]:
                    # Store (Parent -> Improved Offspring)
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
            
        return best_solution, best_fitness, fitness_history

