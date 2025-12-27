import numpy as np
from .tsms_utils import TSMS_Utils, Archive

class WOO_TSMS:
    def __init__(self, func, bounds, pop_size=50, max_generations=1000):
        """
        Wave Optics Optimizer with TSMS Optimization Strategy.
        
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
        self.lam = 5e-4  # Wavelength
        self.d = 1e-2    # Slit width
        
        # Dynamic State
        self.Ua = 0.5    # Initial uncertainty
        
        # TSMS Components
        self.archive_B = Archive(capacity=4*pop_size)  # History archive
        self.archive_A = Archive(capacity=pop_size)    # Inferior archive

    def optimize(self):
        # 1. OBL Initialization
        population, fitness = TSMS_Utils.obl_initialization(self.func, self.bounds, self.pop_size, self.dim)
        
        # Sort
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        best_solution = population[0].copy()
        best_fitness = fitness[0]
        
        fitness_history = []
        
        # TSMS Stage threshold
        rho = int(0.66 * self.max_generations)
        
        # Advantageous Population Params
        MaDim = self.pop_size
        MaDim0 = self.pop_size
        
        # 2. Main Loop
        for t in range(1, self.max_generations + 1):
            
            # --- TSMS Stage Determination ---
            if t < rho:
                # Stage 1: Use Archive B (History)
                pool = self.archive_B.get_candidates(population)
            else:
                # Stage 2: Use Archive A (Inferior)
                pool = self.archive_A.get_candidates(population)
            
            # Dynamic Parameters
            C = 1.0 - t / self.max_generations
            pw = np.sin((1.0 - t/self.max_generations) * np.pi / 2.0)
            
            tau1 = 0.2 * C + 0.8
            tau2_thresh = np.exp(-tau1)
            
            new_population = np.zeros_like(population)
            failed_trials = []
            
            # Calculate Mid index for U1 (Interference)
            T_val = int(abs(round((1.0 - np.sin(np.pi/2.0 * self.Ua)) * self.pop_size)))
            T_val = max(0, min(self.pop_size-1, T_val))
            
            for i in range(self.pop_size):
                X_i = population[i]
                
                # --- Strategy Selection ---
                if np.random.rand() < tau2_thresh:
                    # === Random Strategy (Exploration) ===
                    
                    Rd1 = np.random.randn(self.dim)
                    R1 = np.random.rand()
                    term1 = (R1 * np.abs(Rd1)) ** (C + 3.0)
                    om = term1 + 0.01 * np.random.randn(self.dim)
                    
                    r_rnd = np.random.rand()
                    s_val = np.exp(-2 * r_rnd * (1 - r_rnd))
                    if np.random.rand() < s_val:
                        Z1 = np.ones(self.dim)
                    else:
                        Z1 = np.random.rand(self.dim)
                    
                    # Pick X_mid from pool
                    if len(pool) > 0:
                        mid_idx = np.random.randint(0, len(pool))
                        X_mid = pool[mid_idx]
                    else:
                        mid_idx = np.random.randint(0, T_val)
                        X_mid = population[mid_idx]
                    U1 = (X_mid - X_i) * pw
                    
                    # U2 calculation
                    if i < self.pop_size / 2 and np.random.rand() > C:
                        top_half = max(1, int(self.pop_size/2))
                        if len(pool) > 0:
                            idx1 = np.random.randint(0, len(pool))
                            idx2 = np.random.randint(0, len(pool))
                            U2 = pool[idx1] - pool[idx2]
                        else:
                            idx1 = np.random.randint(0, top_half)
                            idx2 = np.random.randint(0, top_half)
                            U2 = population[idx1] - population[idx2]
                    else:
                        if len(pool) > 0:
                            r_indices = np.random.choice(len(pool), 2, replace=False)
                            U2 = pool[r_indices[0]] - pool[r_indices[1]]
                        else:
                            r_indices = np.random.choice(self.pop_size, 2, replace=False)
                            U2 = population[r_indices[0]] - population[r_indices[1]]
                    
                    # Early Exploration vs Transition
                    w = np.random.rand()
                    if t < self.max_generations / 3:
                        DU1 = w * U1 + (1 - w) * U2
                    else:
                        SU = np.random.rand()
                        DU1 = U1 + SU * U2
                    
                    X_new = X_i + om * Z1 * DU1
                    
                else:
                    # === Intensification Strategy (Exploitation) ===
                    
                    if np.random.rand() > C:
                        # Light Intensity Focus Operator
                        Rd1 = np.random.randn(self.dim)
                        V = Rd1 * (np.sin(np.random.rand(self.dim) * 2 * np.pi))
                        X_new = V * best_solution
                    else:
                        # Beam Modulation
                        r3 = np.random.rand()
                        
                        # Recalculate U2 from pool
                        if len(pool) > 0:
                            r_indices = np.random.choice(len(pool), 3, replace=False)
                            U2 = pool[r_indices[0]] - pool[r_indices[1]]
                        else:
                            r_indices = np.random.choice(self.pop_size, 3, replace=False)
                            U2 = population[r_indices[0]] - population[r_indices[1]]
                        
                        Z2 = np.random.rand(self.dim)
                        om3 = np.random.rand(self.dim) * (1 - r3)
                        
                        # Pick Xr3 and Xr4 from pool
                        if len(pool) > 0:
                            r3_idx = np.random.randint(0, len(pool))
                            r4_idx = np.random.randint(0, len(pool))
                            Xr3 = pool[r3_idx]
                            Xr4 = pool[r4_idx]
                        else:
                            r3_idx = np.random.randint(0, self.pop_size)
                            r4_idx = np.random.randint(0, self.pop_size)
                            Xr3 = population[r3_idx]
                            Xr4 = population[r4_idx]
                        
                        DX = Z2 * om3 * ((best_solution + Xr3)/2.0 - Xr4)
                        X_new = X_i + r3 * np.exp(-1.0) * U2 + DX
                    
                # Boundary Check
                X_new = np.clip(X_new, self.lb, self.ub)
                new_population[i] = X_new
                
            # Evaluation and Selection
            for i in range(self.pop_size):
                f_new = self.func(new_population[i])
                if f_new < fitness[i]:
                    self.archive_B.add([population[i]])
                    population[i] = new_population[i]
                    fitness[i] = f_new
                    if f_new < best_fitness:
                        best_fitness = f_new
                        best_solution = population[i].copy()
                else:
                    failed_trials.append(new_population[i])
            
            # Update Inferior Archive
            if failed_trials:
                self.archive_A.add(failed_trials)
            
            # Sort for next iteration
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            fitness_history.append(best_fitness)
            
            # Update Ua
            self.Ua = self.Ua * 0.99
            
        return best_solution, best_fitness, fitness_history