import numpy as np

class FungalGrowthOptimizer:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000, M=0.6, Ep=0.7):
        """
        Fungal Growth Optimizer (FGO) Implementation.
        
        Args:
            func (callable): Objective function (minimization).
            bounds (list of tuple): [(min, max), ...].
            pop_size (int): Population size (N).
            max_generations (int): Maximum iterations (t_max).
            M (float): Exploration-Exploitation balance parameter (default 0.6).
            Ep (float): Exploration probability/step size parameter (default 0.7).
        """
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

    def optimize(self):
        # 1. Initialization
        # S_i = S_L + r * (S_U - S_L)
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        # Find global best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        fitness_history = []
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # Update global stats
            f_min = np.min(fitness)
            f_max = np.max(fitness)
            denom = f_max - f_min
            if denom == 0: denom = 1e-10
            
            # Exploration Threshold E_r
            # E_r = M + (1 - M) * (1 - t/t_max)
            E_r = self.M + (1.0 - self.M) * (1.0 - t / self.max_generations)
            
            # Growth Rate Base v_i calculations
            # Since min problem, we want better fitness (lower) to have higher growth rate?
            # Text: "Simulate nutrient richness impact on growth rate". High nutrient -> High Growth.
            # Good fitness (Low value) -> High Nutrient.
            # We map fitness to a 'quality' score for v_i calculation.
            # Let's use normalized inverted fitness: q_i = (f_max - f_i) / (f_max - f_min + eps)
            # Then sum_q = sum(q_k)
            # v_i = exp(q_i / (sum_q + eps))
            
            epsilon = 1e-10
            quality = (f_max - fitness) / (denom + epsilon)
            sum_quality = np.sum(quality) + epsilon
            
            v_vec = np.exp(quality / sum_quality)
            
            new_population = np.zeros_like(population)
            
            for i in range(self.pop_size):
                # Normalized fitness p_i for switching rule
                # p_i = (f(S_i) - min(f)) / (max(f) - min(f) + eps)
                # Best individual has p_i = 0. Worst has p_i = 1.
                p_i = (fitness[i] - f_min) / (denom + epsilon)
                
                S_i = population[i]
                
                # Switch Rule
                if p_i < E_r:
                    # === Exploration Phase ===
                    # Behaviors: Tip Growth, Branching, Spore Germination
                    
                    r9 = np.random.rand()
                    r10 = np.random.rand()
                    
                    # 1. Hyphal Tip Growth (Prob ~ 0.5)
                    if r9 < r10:
                        # Adjusted Growth Rate
                        # v_adj = v_i * r1 * (3 * (1 - t/t_max))
                        r1 = np.random.rand()
                        decay = 3.0 * (1.0 - t / self.max_generations)
                        v_adj = v_vec[i] * r1 * decay
                        
                        # Direction D
                        # Select random S_a, S_c (different from i?)
                        # Text says "Randomly select two solutions".
                        idxs = np.random.choice(self.pop_size, 2, replace=False) # Simplification: might pick i, but rare
                        S_a = population[idxs[0]]
                        S_c = population[idxs[1]]
                        
                        D = (S_a - S_c) * v_adj
                        
                        # Update S_new = S_i + D * [R_j < r3]
                        # Partial update
                        r3 = np.random.rand()
                        mask = (np.random.rand(self.dim) < r3).astype(float)
                        
                        S_new = S_i + D * mask
                        
                    else:
                        r7 = np.random.rand()
                        
                        # 2. Hyphal Branching (Prob ~ 0.25)
                        if r7 < 0.5:
                            # D_branch1 = S_a - S_b
                            # D_branch2 = S* - S_c
                            idxs = np.random.choice(self.pop_size, 3, replace=False)
                            S_a = population[idxs[0]]
                            S_b = population[idxs[1]]
                            S_c = population[idxs[2]]
                            
                            D_b1 = S_a - S_b
                            D_b2 = best_solution - S_c
                            
                            r5 = np.random.rand()
                            if r5 > 0.5:
                                D_branch = D_b1
                            else:
                                D_branch = D_b2
                                
                            # Update S_new = S_i + r6 * D_branch * [R_j < r3]
                            r6 = np.random.rand()
                            r3 = np.random.rand()
                            mask = (np.random.rand(self.dim) < r3).astype(float)
                            
                            S_new = S_i + r6 * D_branch * mask
                            
                        # 3. Spore Germination (Prob ~ 0.25)
                        else:
                            # S_spore = (S* + S_a + S_b) / 3
                            idxs = np.random.choice(self.pop_size, 2, replace=False)
                            S_a = population[idxs[0]]
                            S_b = population[idxs[1]]
                            
                            S_spore = (best_solution + S_a + S_b) / 3.0
                            
                            # v_adj reuse or recalculate? Text implies same logic.
                            r1 = np.random.rand()
                            decay = 3.0 * (1.0 - t / self.max_generations)
                            v_adj = v_vec[i] * r1 * decay
                            
                            # S_g in {-1, 1}
                            S_g = 1.0 if np.random.rand() < 0.5 else -1.0
                            
                            # U vector (dimension selector). "Inherit parent characteristics".
                            # Let's assume random binary mask like crossover.
                            # Text: "U is dimension selection vector".
                            # Formula: S_new = S_spore + S_g * v_adj * |S_spore - S_i|
                            # Note: The U term usually applies to crossover. The formula in image 3
                            # S_new = S_spore + S_g * v_adj * |S_spore - S_i|
                            # Where did U go?
                            # Ah, looking at Image 3 equation:
                            # S(t+1) = S_spore + S_g * v_adj * |S_spore - S_i|
                            # Wait, the text below says "U is dimension selection vector".
                            # Maybe the equation implies: S_new = S_spore ... for some dims, and something else for others?
                            # Or maybe the term `v_adj` implies the `v_i^{adj}` which was a scalar?
                            # Let's look closely at image 3 again.
                            # S_i^{t+1} = S_spore + S_g * v_i^{adj} * |S_spore - S_i^t|
                            # It doesn't show U in the equation line.
                            # But text says U is there.
                            # Standard interpretation: The update is applied to all dimensions?
                            # Let's assume full dimension update for Spore Germination based on equation.
                            
                            S_new = S_spore + S_g * v_adj * np.abs(S_spore - S_i)
                            
                else:
                    # === Exploitation Phase (Chemotropism) ===
                    # State 1 (Towards Nutrient)
                    
                    # Nutrient Distribution Coefficient ni
                    # if t < t_max/2: ni = r7 (random)
                    # else: ni = f(S_i) / (sum f + eps) -> Use quality q_i / sum_q
                    
                    if t < self.max_generations / 2:
                        ni = np.random.rand()
                    else:
                        ni = quality[i] / sum_quality
                        
                    # Normalized ni? Formula says ni^{norm}. Maybe ni is already normalized by sum?
                    # Let's use ni as calculated.
                    
                    # D_chem1 = r5 * (beta * S* + (1-beta)*S_a - S_i)
                    idxs = np.random.choice(self.pop_size, 1, replace=False)
                    S_a = population[idxs[0]]
                    
                    beta = 1.0 if np.random.rand() < 0.5 else -1.0 # Wait, beta in {1, -1}? Or [0,1]?
                    # Image 5: "beta in {1, -1}". Text says "(-1 means far from optimal)".
                    # Formula: beta * S* + (1-beta) * S_a
                    # If beta=1: 1*S* + 0*S_a = S*
                    # If beta=-1: -1*S* + 2*S_a ... This seems like "away from best".
                    # Let's follow formula.
                    
                    r5 = np.random.rand()
                    D_chem1 = r5 * (beta * best_solution + (1.0 - beta) * S_a - S_i)
                    
                    # S_new = S_i + ni * D_chem1 + r9 * Ep * [r11 > r13]
                    # Note: ni * D_chem1 provides the direction.
                    # The last term is random walk? r9 * Ep * Mask
                    
                    r9 = np.random.rand()
                    r11 = np.random.rand()
                    r13 = np.random.rand()
                    
                    # Mask for random walk (usually scalar condition applied to all or vector?)
                    # [r11 > r13] notation usually means Iverson bracket. 1 if true, 0 if false.
                    # Applied to vector? Usually scalar check, applied to update.
                    
                    walk_trigger = 1.0 if r11 > r13 else 0.0
                    
                    S_new = S_i + ni * D_chem1 + r9 * self.Ep * walk_trigger * (np.random.rand(self.dim) * 2 - 1) 
                    # Note: Formula says "r9 * Ep * [..]". Usually assumes random direction or just magnitude added?
                    # Given vectors, adding a scalar "r9*Ep" doesn't make sense unless broadcast.
                    # Assuming random perturbation vector in range [-1, 1] scaled by Ep.
                    
                
                # Boundary Check
                S_new = np.clip(S_new, self.lb, self.ub)
                new_population[i] = S_new
            
            # Selection (Greedy?)
            # The algorithm description doesn't explicitly state "Greedy Selection" vs "Direct Replacement".
            # Bio-algos usually evaluate and keep best (Greedy) or just replace (Generational).
            # "Comparison new solution vs original... keep better" is standard.
            # Let's assume Greedy Selection.
            
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                if f_new < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
            
            fitness_history.append(best_fitness)
            
        return best_solution, best_fitness, fitness_history

