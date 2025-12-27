import numpy as np

class WaveOpticsOptimizer:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        """
        Wave Optics Optimizer (WOO) Implementation.
        
        Args:
            func (callable): Objective function.
            bounds (list of tuple): [(min, max), ...].
            pop_size (int): Population size (N).
            max_generations (int): Maximum iterations (MaxIt).
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_generations = max_generations
        
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.range_width = self.ub - self.lb
        
        # Physics Constants / Parameters
        self.lam = 5e-4 # Wavelength
        self.d = 1e-2   # Slit width
        
        # Dynamic State
        self.Ua = 0.5   # Initial uncertainty
        
    def optimize(self):
        # 1. Initialization
        population = self.lb + self.range_width * np.random.rand(self.pop_size, self.dim)
        fitness = np.array([self.func(ind) for ind in population])
        
        # Sort
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        best_solution = population[0].copy()
        best_fitness = fitness[0]
        
        fitness_history = []
        
        # Advantageous Population Params
        MaDim = self.pop_size
        MaDim0 = self.pop_size
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # Dynamic Parameters
            # C from 1 to 0
            C = 1.0 - t / self.max_generations
            
            # pw: Interference coefficient
            # Formula: sin( (-e^(It/MaxIt?) + 1) * pi/2 )?
            # Text image 5: pw = sin( (-e^(It) + 1) * pi/2 ). 
            # If It is raw iteration, e^It explodes. Likely normalized It/MaxIt.
            # Assuming pw decays or oscillates. 
            # Let's try: pw = sin( (-exp(t/MaxIt) + 2) * pi/2 ) to keep inside range?
            # Or standard decay: pw = sin( (1 - t/MaxIt) * pi/2 )?
            # Let's approximate based on visual: decays from 1 to 0.
            pw = np.sin((1.0 - t/self.max_generations) * np.pi / 2.0)
            
            # tau1, tau2 for Uask and Strategy Selection
            # tau1 = 0.2*C + 0.8
            # tau2 = exp(tau1) ? No, image 6 formula for Uask involves tau2 > rand.
            # Image 6 bottom: tau1 = 0.2C + 0.8. tau2 = e^tau1? No, e^tau1 > 1 usually (if tau1 ~ 0.8-1.0).
            # Maybe tau2 = e^(-tau1)? Or scaled?
            # Let's assume tau2 is a threshold probability.
            # Step 3 in Image 6: "If tau2 > rand execute Random...".
            # If tau2 > 1, always Random.
            # Let's assume tau2 is dynamic, maybe around 0.5-0.9?
            # Let's look at Uask formula: tau2 = e^{tau1}. This is > 1.
            # Maybe the condition is something else or tau2 is used differently.
            # Let's assume Strategy Selection Threshold is C (common in algos) or fixed.
            # Actually Image 6 says "tau2 > rand".
            # Let's try tau2 = exp(-tau1) which would be < 1.
            tau1 = 0.2 * C + 0.8
            tau2_thresh = np.exp(-tau1) # Guessing the minus sign
            
            new_population = np.zeros_like(population)
            
            # Calculate Mid index for U1 (Interference)
            # T calculation (Image 5)
            # T = |round( (1 - sin(pi/2 * Ua)) * N )|
            # Ua starts at 0.5?
            T_val = int(abs(round((1.0 - np.sin(np.pi/2.0 * self.Ua)) * self.pop_size)))
            T_val = max(0, min(self.pop_size-1, T_val))
            
            # Mid index range? Image 5: Mid = [T+1, round(N - ...)]
            # Simplified: Select from top portion?
            # Let's define Mid population as range [0, T_val] or [T_val, N]?
            # U1 = (X_{r<-Mid} - X_i).
            # Let's assume Mid means "Better" solutions (Top T).
            
            # Compute U2 for all i (Image 5)
            # U2 depends on i < N/2 and rand > C
            
            for i in range(self.pop_size):
                X_i = population[i]
                
                # --- Strategy Selection ---
                # "If tau2 > rand" -> Random Strategy.
                # Assuming tau2_thresh calculated above.
                
                if np.random.rand() < tau2_thresh:
                    # === Random Strategy (Exploration) ===
                    
                    # om calculation (Image 2)
                    # om = (R1 * |Rd1|)^(C+3) + ...
                    # Rd1 usually random vector [-1, 1] or normal.
                    Rd1 = np.random.randn(self.dim)
                    R1 = np.random.rand()
                    
                    # Term 1
                    term1 = (R1 * np.abs(Rd1)) ** (C + 3.0)
                    
                    # rd1, mr? Image 2 text is blurry. "rd1 / mr".
                    # Let's use simple random perturbation.
                    om = term1 + 0.01 * np.random.randn(self.dim)
                    
                    # Z1 (Image 2)
                    # Z1 = R2 < s ? exp(...) : ...
                    # Formula: R2 < s = exp(-2*r1*(1-r1)).
                    # Looks like a condition. s is calculated.
                    r_rnd = np.random.rand()
                    s_val = np.exp(-2 * r_rnd * (1 - r_rnd))
                    if np.random.rand() < s_val:
                        Z1 = np.ones(self.dim) # Placeholder, text is cutoff
                    else:
                        Z1 = np.random.rand(self.dim)
                        
                    # DU1 calculation (Image 2)
                    # U1 = (X_mid - X_i) * pw
                    # Pick X_mid
                    if T_val > 0:
                        mid_idx = np.random.randint(0, T_val)
                    else:
                        mid_idx = 0
                    X_mid = population[mid_idx]
                    U1 = (X_mid - X_i) * pw
                    
                    # U2 calculation (Image 5)
                    # If i < N/2 and rand > C: X_{r1} - X_{r2} (Top half)
                    # Else: X_{r2} - X_{r3}
                    r_indices = np.random.choice(self.pop_size, 3, replace=False)
                    if i < self.pop_size / 2 and np.random.rand() > C:
                        # Use top half randoms?
                        top_half = max(1, int(self.pop_size/2))
                        idx1 = np.random.randint(0, top_half)
                        idx2 = np.random.randint(0, top_half)
                        U2 = population[idx1] - population[idx2]
                    else:
                        U2 = population[r_indices[0]] - population[r_indices[1]]
                        
                    # DU1 formula Image 2
                    # w = Ir * e^(-R3^(C+3))? Ir is "Intensity coefficient"?
                    # Image 2 bottom: Ir = e^{|i| * ...}.
                    # Let's approximate w as weight.
                    w = np.random.rand()
                    
                    # Early Exploration vs Transition
                    # Threshold: 1/3 MaxIt?
                    if t < self.max_generations / 3:
                        DU1 = w * U1 + (1 - w) * U2
                    else:
                        # SU = ...
                        SU = np.random.rand()
                        DU1 = U1 + SU * U2
                        
                    # Final Update Random
                    # X_new = X_i + om * Z1 * DU1
                    X_new = X_i + om * Z1 * DU1
                    
                else:
                    # === Intensification Strategy (Exploitation) ===
                    
                    # Image 3
                    # Light Intensity Focus Operator (rand > C)
                    if np.random.rand() > C:
                        # V * BestX
                        # V formula Image 3: Rd1 * sqrt(2) ... sin/cos soup
                        # Simplified V: Random modulation around 1
                        # V ~ Normal(1, 0.1) or similar?
                        # Let's implement full formula if possible or approx.
                        # V = Rd1 * sqrt(2) * sqrt(abs(sin...)) / ...
                        # Approximating V as random vector scaling
                        Rd1 = np.random.randn(self.dim)
                        V = Rd1 * (np.sin(np.random.rand(self.dim) * 2 * np.pi))
                        
                        X_new = V * best_solution
                    else:
                        # Beam Modulation
                        # X_new = X_i + r3 * exp(...) * U2 + DX
                        # DX = Z2 * om3 * ( (Best+Xr3)/2 - Xr4 )
                        
                        r3 = np.random.rand()
                        
                        # Recalculate U2
                        r_indices = np.random.choice(self.pop_size, 3, replace=False)
                        U2 = population[r_indices[0]] - population[r_indices[1]]
                        
                        # DX
                        Z2 = np.random.rand(self.dim)
                        om3 = np.random.rand(self.dim) * (1 - r3) # Approx from Image 3
                        
                        Xr3 = population[r_indices[1]]
                        Xr4 = population[r_indices[2]]
                        
                        DX = Z2 * om3 * ( (best_solution + Xr3)/2.0 - Xr4 )
                        
                        X_new = X_i + r3 * np.exp(-1.0) * U2 + DX # approx exp arg
                        
                # Boundary Check
                X_new = np.clip(X_new, self.lb, self.ub)
                new_population[i] = X_new
                
            # Evaluation and Selection
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                if f_new < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = new_population[i].copy()
                        
            # Sort for next iteration
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            fitness_history.append(best_fitness)
            
            # 3. Population Adjustment (Uask update)
            # Image 6 bottom
            # Uask = ...
            # Depends on Advantageous Population delta.
            # Simplified: Decay Ua
            # If tau2 > rand: Ua = tau1 * L1 + ...
            # This logic seems to maintain diversity.
            # I will implement a simple decay for Ua to simulate convergence.
            self.Ua = self.Ua * 0.99
            
        return best_solution, best_fitness, fitness_history

