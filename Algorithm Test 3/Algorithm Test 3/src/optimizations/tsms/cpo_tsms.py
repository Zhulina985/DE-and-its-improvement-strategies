import numpy as np
from .tsms_utils import TSMS_Utils, Archive

class CPO_TSMS:
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
        
        # TSMS Components
        self.archive_B = Archive(capacity=4*pop_size) # History
        self.archive_A = Archive(capacity=pop_size)   # Inferior
        
    def optimize(self):
        # 1. OBL Initialization
        population, fitness = TSMS_Utils.obl_initialization(self.func, self.bounds, self.pop_size, self.dim)
        
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        current_pop_size = self.pop_size
        
        rho = int(0.66 * self.max_generations) # Stage threshold
        
        for t in range(1, self.max_generations + 1):
            
            # --- CPO Reduction ---
            step = int(t / (self.max_generations / self.T))
            if step >= self.T: step = self.T - 1
            target_size = int(max(self.N_min, self.pop_size - step * ((self.pop_size - self.N_min) / self.T)))
            if target_size < current_pop_size:
                # Remove worst, add to Archive A (Inferior)?
                # Actually Archive A stores failed trials.
                population = population[:target_size]
                fitness = fitness[:target_size]
                current_pop_size = target_size
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx].copy()
            X_CP = best_solution
            
            # --- TSMS Stage Determination ---
            if t < rho:
                # Stage 1: Use Archive B (History)
                pool = self.archive_B.get_candidates(population)
            else:
                # Stage 2: Use Archive A (Inferior)
                pool = self.archive_A.get_candidates(population)
                
            ratio = t / self.max_generations
            if ratio >= 1.0: ratio = 0.9999
            gamma_t = 2 * np.random.rand() * ((1 - ratio) ** ratio)
            sum_fitness = np.sum(np.abs(fitness)) + 1e-10
            
            new_population = np.zeros_like(population)
            failed_trials = []
            
            for i in range(current_pop_size):
                X_i = population[i]
                
                # Pick r1 from Population (Standard)
                candidates = [idx for idx in range(current_pop_size) if idx != i]
                if len(candidates) < 1: r1 = i
                else: r1 = np.random.choice(candidates)
                
                # Pick r2 from Pool (Archive strategy)
                # Standard CPO uses X_r2 from population. TSMS substitutes it.
                if len(pool) > 0:
                    r2_idx = np.random.randint(0, len(pool))
                    X_r2 = pool[r2_idx]
                else:
                    X_r2 = population[np.random.choice(candidates)]
                
                # Pick r3 if needed
                if len(pool) > 0:
                    r3_idx = np.random.randint(0, len(pool))
                    X_r3 = pool[r3_idx]
                else:
                    X_r3 = population[np.random.choice(candidates)]
                
                # --- CPO Logic Modified ---
                # Substitute population[r2] -> X_r2, population[r3] -> X_r3
                
                tau6, tau7, tau10 = np.random.rand(), np.random.rand(), np.random.rand()
                
                if np.random.rand() < 0.5:
                    if tau6 < tau7:
                        # Visual: y_i = (X_i + X_r) / 2
                        y_i = (X_i + population[r1]) / 2.0 # Keep r1 from pop
                        I1 = np.random.normal()
                        tau2 = np.random.rand()
                        new_pos = X_CP + I1 * (y_i - tau2 * X_i)
                    else:
                        # Sound: U1 * (y_i - sign(r1-r2)*Xi)
                        y_i = (X_i + population[r1]) / 2.0
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        # Use X_r2 from Archive
                        diff_sign = np.sign(population[r1] - X_r2)
                        new_pos = X_i + U1 * (y_i - diff_sign * X_i)
                else:
                    S_i = np.exp(fitness[i] / sum_fitness)
                    delta = 1.0 if np.random.rand() > 0.5 else -1.0
                    
                    if tau10 < self.Tf:
                        # Odor: X_r2 - X_r3 used
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        tau3 = np.random.rand()
                        term2 = population[r1] + S_i * (X_r2 - X_r3) - \
                                tau3 * delta * gamma_t * S_i
                        new_pos = (1 - U1) * X_i + U1 * term2
                    else:
                        # Physical
                        tau4, tau5 = np.random.rand(), np.random.rand()
                        v_next, v_curr = population[r1], X_i
                        m_i = np.exp(fitness[i] / sum_fitness)
                        F_i = tau6 * m_i * (v_next - v_curr)
                        factor = self.alpha * (1 - tau4) + tau4
                        term_mid = delta * X_CP - X_i
                        term_last = tau5 * delta * gamma_t * F_i
                        new_pos = X_CP + factor * term_mid - term_last
                        
                new_pos = np.clip(new_pos, lb, ub)
                new_population[i] = new_pos
                
            # Evaluation
            for i in range(current_pop_size):
                f_new = self.func(new_population[i])
                
                if f_new < fitness[i]:
                    # Success
                    self.archive_B.add([population[i]]) # Add good parent to History
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                else:
                    # Failure
                    failed_trials.append(new_population[i])
            
            # Update Inferior Archive
            if failed_trials:
                self.archive_A.add(failed_trials)
                
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

