import numpy as np

class CrestedPorcupineOptimizer:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        """
        Crested Porcupine Optimizer (CPO) Implementation.
        
        Args:
            func (callable): Objective function.
            bounds (list of tuple): [(min, max), ...].
            pop_size (int): Initial Population size (N').
            max_generations (int): Maximum iterations (T_max).
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        # Parameters (based on text descriptions and standard defaults for this algo)
        self.T = 2  # Cycles for population reduction
        self.alpha = 0.2  # Convergence rate factor
        self.Tf = 0.8  # Balance parameter between Odor and Physical
        self.N_min = max(5, int(self.pop_size * 0.1)) # Minimum population size estimate
        
    def optimize(self):
        # 1. Initialization
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        population = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
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
            # N(t) formula implementation
            # Factor determines which 'step' of reduction we are in [0, T-1]
            step = int(t / (self.max_generations / self.T))
            if step >= self.T: 
                step = self.T - 1
            
            # Theoretical new size
            target_size = int(max(self.N_min, self.pop_size - step * ((self.pop_size - self.N_min) / self.T)))
            
            # If we need to reduce
            if target_size < current_pop_size:
                # We assume population is already sorted at start of loop or end of prev
                # Keep top 'target_size'
                population = population[:target_size]
                fitness = fitness[:target_size]
                current_pop_size = target_size
            
            # Update best just in case (though we keep it sorted)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = population[best_idx].copy()
                
            X_CP = best_solution
            
            # Pre-calculate common factors
            # gamma_t for Odor/Physical
            # Power term: (1 - t/Tmax)^(t/Tmax)
            # Avoid divide by zero or negative base if t=Tmax
            ratio = t / self.max_generations
            if ratio >= 1.0: ratio = 0.9999
            gamma_t = 2 * np.random.rand() * ((1 - ratio) ** ratio)
            
            # Sum of fitness for S_i calculation (using abs to avoid sign issues if fit < 0?)
            # Formula says sum(f). If f is minimized, this is just a normalization factor.
            # We add epsilon to avoid div by zero.
            sum_fitness = np.sum(np.abs(fitness)) + 1e-10
            
            new_population = np.zeros_like(population)
            new_fitness = np.zeros(current_pop_size)
            
            # Generate random thresholds for decision making
            # Assuming random switch between Exploration/Exploitation for the population
            # OR individual based. The text says "Randomly generate... judge". 
            # We'll do individual based as it's common in metaheuristics.
            
            for i in range(current_pop_size):
                X_i = population[i]
                
                # Random candidates
                candidates = [idx for idx in range(current_pop_size) if idx != i]
                if len(candidates) < 3: # Fallback for very small pop
                    r_indices = [np.random.randint(0, current_pop_size) for _ in range(3)]
                else:
                    r_indices = np.random.choice(candidates, 3, replace=False)
                r1, r2, r3 = r_indices[0], r_indices[1], r_indices[2]
                
                # Random vars
                tau6 = np.random.rand()
                tau7 = np.random.rand()
                tau10 = np.random.rand()
                
                # Decision: Exploration vs Exploitation
                # We use a simple 50/50 probability trigger or based on convergence?
                # The paper doesn't explicitly state the master switch formula in the provided text.
                # However, many bio-inspired algos use rand < 0.5.
                
                if np.random.rand() < 0.5:
                    # === EXPLORATION PHASE ===
                    
                    # Strategy 1: Visual Defense (tau6 < tau7)
                    if tau6 < tau7:
                        # y_i = (X_i + X_r) / 2. Let's pick a random r different from i
                        r = r1 
                        y_i = (X_i + population[r]) / 2.0
                        
                        I1 = np.random.normal() # Normal dist
                        tau2 = np.random.rand()
                        
                        new_pos = X_CP + I1 * (y_i - tau2 * X_i)
                        
                    # Strategy 2: Sound Defense (tau6 >= tau7)
                    else:
                        # y_i = (X_i + X_r) / 2
                        r = r1
                        y_i = (X_i + population[r]) / 2.0
                        
                        # U1 is binary vector {0, 1}
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        
                        # sign(X_r1 - X_r2)
                        diff_sign = np.sign(population[r1] - population[r2])
                        
                        new_pos = X_i + U1 * (y_i - diff_sign * X_i)
                
                else:
                    # === EXPLOITATION PHASE ===
                    
                    # Common calc for Exploitation
                    # S_i = exp( f(X_i) / sum_fitness )
                    # Note: If minimizing, large fitness is bad. 
                    # But the formula is literal.
                    S_i = np.exp(fitness[i] / sum_fitness)
                    
                    # delta: Search direction control parameter. +/- 1? 
                    # Assuming random sign.
                    delta = 1.0 if np.random.rand() > 0.5 else -1.0
                    
                    # Strategy 3: Odor Defense (tau10 < Tf)
                    if tau10 < self.Tf:
                        U1 = (np.random.rand(self.dim) < 0.5).astype(float)
                        tau3 = np.random.rand()
                        
                        # X_new = (1-U1)*X_i + U1*(X_r1 + S_i*(X_r2 - X_r3) - tau3*delta*gamma_t*S_i)
                        # Note: The last term tau3... is scalar subtraction? Or vector?
                        # Likely scalar applied to all dims or broadcast.
                        
                        term2 = population[r1] + S_i * (population[r2] - population[r3]) - \
                                tau3 * delta * gamma_t * S_i
                                
                        new_pos = (1 - U1) * X_i + U1 * term2
                        
                    # Strategy 4: Physical Attack (tau10 >= Tf)
                    else:
                        tau4 = np.random.rand()
                        tau5 = np.random.rand()
                        
                        # F_i calculation
                        # v_i^{t+1} = X_r (Predictor position? Formula says x_r^t)
                        # v_i^t ... current position X_i? Or stored velocity?
                        # Given no memory in class, assume v_i^t = X_i
                        v_next = population[r1] # Random individual
                        v_curr = X_i
                        
                        m_i = np.exp(fitness[i] / sum_fitness)
                        F_i = tau6 * m_i * (v_next - v_curr)
                        
                        # X_new = X_CP + (alpha(1-tau4) + tau4)*(delta*X_CP - X_i) - tau5*delta*gamma_t*F_i
                        factor = self.alpha * (1 - tau4) + tau4
                        term_mid = delta * X_CP - X_i
                        term_last = tau5 * delta * gamma_t * F_i
                        
                        new_pos = X_CP + factor * term_mid - term_last

                # Boundary clamping
                new_pos = np.clip(new_pos, lb, ub)
                new_population[i] = new_pos
            
            # Evaluate New Population
            # (In standard CPO, we greedily select or just replace? 
            #  Text "Update rules" implies generating X(t+1).
            #  "Evaluate new fitness, keep better, update global best". 
            #  This implies greedy selection between X_i and New_Pos.)
            
            for i in range(current_pop_size):
                f_new = self.func(new_population[i])
                
                if f_new < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            # Sort population for next iteration (needed for X_CP and reduction)
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history
