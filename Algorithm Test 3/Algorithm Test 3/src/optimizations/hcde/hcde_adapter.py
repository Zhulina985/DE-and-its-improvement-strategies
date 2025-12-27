"""
HCDE Strategy Adapter for Non-DE Algorithms
Implements HCDE (Hierarchically Controlled Differential Evolution) strategies:
1. Entropy-based diversity measurement
2. Hybrid perturbation (Gaussian + Cauchy)
3. Multi-level archive mechanism
4. Bi-stage parameter adaptation
"""

import numpy as np
try:
    from scipy.linalg import eigh
except ImportError:
    from numpy.linalg import eigh


class HCDEAdapter:
    """Adapter for applying HCDE strategies to non-DE algorithms"""
    
    def __init__(self, dim, pop_size, max_generations, bounds):
        self.dim = dim
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.bounds = np.array(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        
        self.nfes = 0
        self.nfes_max = max_generations * pop_size
        
        # Parameter adaptation memory
        self.H = 4
        self.mu_param = np.ones(self.H) * 0.5
        self.memory_index = 0
        self.S_param = []
        
        # Multi-level archives
        self.archive_A = []  # Elite archive
        self.archive_B = []  # Promising non-elite archive
        self.racA = 1.5  # Archive A capacity ratio
        self.racB = 0.5  # Archive B capacity ratio
        self.max_A = int(self.racA * pop_size)
        self.max_B = int(self.racB * pop_size)
        
        # Stagnation detection
        self.count = np.zeros(pop_size)
        self.count_threshold = 2 * dim  # ct = 2*D
        
        # Diversity and perturbation parameters
        self.xi = 0.001  # Diversity threshold
        self.lambda_decay = 2.0  # Decay rate (2 for high-dim, 5 for low-dim)
        self.a_min = 0.1
        self.a_max = 0.5
        
        # Stage threshold
        self.l = int(self.nfes_max * 0.2)
    
    def compute_entropy_diversity(self, population):
        """
        Compute entropy-based diversity measure
        Based on Equations 15-19 from HCDE paper
        """
        PS, D = population.shape
        
        # Normalize population for each dimension
        normalized_pop = np.zeros_like(population)
        H_norm = np.zeros(D)
        
        for d in range(D):
            x_d = population[:, d]
            min_d = np.min(x_d)
            max_d = np.max(x_d)
            
            if max_d == min_d:
                H_norm[d] = 0.0
                continue
            
            # Normalization (Equation 15)
            normalized_pop[:, d] = (x_d - min_d) / (max_d - min_d)
            x_d_norm = normalized_pop[:, d]
            
            # Compute IQR for Freedman-Diaconis rule (Equation 16)
            Q1 = np.percentile(x_d_norm, 25)
            Q3 = np.percentile(x_d_norm, 75)
            IQR_d = Q3 - Q1
            
            # Add numerical stability check for very small IQR
            if IQR_d <= 1e-10 or IQR_d == 0:
                H_norm[d] = 0.0
                continue
            
            # Bin count using Freedman-Diaconis rule (Equation 17)
            range_d = np.max(x_d_norm) - np.min(x_d_norm)
            if range_d == 0:
                B_d = 1
            else:
                # Calculate B_d with safety limits to prevent memory issues
                denominator = 2 * IQR_d * PS**(-1/3)
                if denominator <= 1e-10:
                    B_d = min(PS, 100)  # Use reasonable upper limit
                else:
                    B_d = max(1, int(range_d / denominator))
                    # Cap B_d to prevent excessive memory allocation
                    B_d = min(B_d, max(PS, 100))
            
            # Compute probability distribution
            hist, bin_edges = np.histogram(x_d_norm, bins=B_d, range=(0, 1))
            hist = hist + 1e-10  # Avoid zero probabilities
            p_k = hist / np.sum(hist)
            
            # Normalized entropy (Equation 18)
            entropy = -np.sum(p_k * np.log2(p_k + 1e-10))
            if B_d > 1:
                H_norm[d] = entropy / np.log2(B_d)
            else:
                H_norm[d] = 0.0
        
        # Population diversity (Equation 19)
        D_div = np.mean(H_norm)
        
        return D_div, H_norm
    
    def get_adaptive_parameter(self, base_value=0.5):
        """
        Bi-stage parameter adaptation using logistic and Cauchy distributions
        Based on Equations 6-8 from HCDE paper
        """
        r = np.random.randint(0, self.H)
        mu_r = self.mu_param[r]
        
        if self.nfes < self.l:
            # Stage 1: Early stage - use logistic-like distribution
            # Simplified: use uniform with higher mean
            param = mu_r + 0.1 * np.random.rand()
            param = np.clip(param, 0.6, 1.0)  # CR in [0.6, 1] for early stage
        else:
            # Stage 2: Later stage - use Cauchy distribution
            param = np.random.standard_cauchy() * 0.1 + mu_r
            param = np.clip(param, 0.0, 1.0)  # CR in [0, 1] for later stage
        
        return param
    
    def hybrid_perturbation(self, X_i, X_gbest, population, ct_i):
        """
        Hybrid perturbation combining Gaussian and Cauchy distributions
        Based on Equations 20-24 from HCDE paper
        """
        # Decay term (Equation 21)
        decay = np.exp(-self.lambda_decay * (self.nfes / self.nfes_max))
        
        # Adaptive coefficient (Equation 20)
        if ct_i <= self.count_threshold:
            a = self.a_min
        else:
            # Linear interpolation between a_min and a_max
            progress = min(1.0, (ct_i - self.count_threshold) / self.count_threshold)
            a = self.a_min + (self.a_max - self.a_min) * progress
        
        # Gaussian term: (X_i - X_gbest)^2 * exp(-(X_i - X_gbest)^2 / 2)
        diff = X_i - X_gbest
        # Clip diff to prevent numerical overflow in exponential
        diff_squared = np.clip(diff**2, 0, 100)  # Cap at 100 to prevent overflow
        gaussian_term = diff_squared * np.exp(-diff_squared / 2.0)
        
        # Cauchy term: C(0, 1) - clip to prevent extreme values
        cauchy_term = np.random.standard_cauchy(self.dim)
        cauchy_term = np.clip(cauchy_term, -1e6, 1e6)  # Prevent extreme outliers
        
        # Average direction vector (Equation 23)
        if len(population) > 1:
            # Select N random individuals (N = min(10, pop_size))
            N = min(10, len(population))
            indices = np.random.choice(len(population), N, replace=False)
            A_dir = np.mean([population[j] - X_i for j in indices], axis=0)
        else:
            A_dir = np.zeros(self.dim)
        
        gamma = np.random.randn()  # Random scaling for direction
        direction_term = gamma * A_dir
        
        # Hybrid perturbation (Equation 22, 24)
        perturb = gaussian_term + cauchy_term + direction_term
        
        # New position (Equation 20)
        X_new = decay * X_i + a * perturb
        
        return X_new
    
    def update_archives(self, population, fitness, best_solution, best_fitness):
        """
        Update multi-level archives A and B
        Archive A: Elite solutions
        Archive B: Promising non-elite solutions
        """
        PS = len(population)
        
        # Update archive size limits if population size changed
        if PS != self.pop_size:
            self.pop_size = PS
            self.max_A = int(self.racA * PS)
            self.max_B = int(self.racB * PS)
            # Trim archives if they exceed new limits
            if len(self.archive_A) > self.max_A:
                self.archive_A = self.archive_A[:self.max_A]
            if len(self.archive_B) > self.max_B:
                self.archive_B = self.archive_B[:self.max_B]
        
        avg_fitness = np.mean(fitness)
        
        # Update Archive A (elite)
        elite_indices = np.where(fitness < best_fitness * 1.1)[0]  # Top 10% better than best
        for idx in elite_indices:
            if len(self.archive_A) < self.max_A:
                self.archive_A.append(population[idx].copy())
            else:
                # Replace worst in archive
                if len(self.archive_A) > 0:
                    worst_idx = np.argmax([self._archive_fitness(x) for x in self.archive_A])
                    self.archive_A[worst_idx] = population[idx].copy()
        
        # Update Archive B (promising non-elite)
        promising_indices = np.where((fitness < avg_fitness) & (fitness >= best_fitness * 1.1))[0]
        for idx in promising_indices:
            if len(self.archive_B) < self.max_B:
                self.archive_B.append(population[idx].copy())
            else:
                # Replace worst in archive
                if len(self.archive_B) > 0:
                    worst_idx = np.argmax([self._archive_fitness(x) for x in self.archive_B])
                    self.archive_B[worst_idx] = population[idx].copy()
    
    def _archive_fitness(self, x):
        """Helper to estimate fitness for archive management"""
        # Simple distance-based estimate
        return np.sum(x**2)
    
    def select_from_archive(self, archive_type='A'):
        """
        Select individual from archive using timestamp mechanism
        """
        if archive_type == 'A':
            archive = self.archive_A
        else:
            archive = self.archive_B
        
        if len(archive) == 0:
            return None
        
        # Simple random selection (timestamp mechanism simplified)
        idx = np.random.randint(0, len(archive))
        return archive[idx].copy()
    
    def update_stagnant_individuals(self, population, fitness, best_solution, best_fitness):
        """
        Detect and update stagnant individuals using entropy-based diversity
        Based on Algorithm 2 and Equations 20-24
        """
        updated_population = population.copy()
        updated_fitness = fitness.copy()
        
        # Ensure count array matches population size
        pop_size = len(population)
        if len(self.count) != pop_size:
            self.count = np.zeros(pop_size)
        
        # Compute diversity
        D_div, _ = self.compute_entropy_diversity(population)
        
        # Update stagnation counters (Algorithm 2)
        for i in range(pop_size):
            if fitness[i] >= best_fitness:  # No improvement
                self.count[i] += 1
            else:
                self.count[i] = 0
        
        # Find stagnant individuals
        # Stagnant if: D_div > xi AND ct > count_threshold
        stagnant_indices = np.where((D_div > self.xi) & (self.count > self.count_threshold))[0]
        
        if len(stagnant_indices) > 0:
            for idx in stagnant_indices:
                ct_i = self.count[idx]
                
                # Apply hybrid perturbation
                new_pos = self.hybrid_perturbation(
                    population[idx], 
                    best_solution, 
                    population, 
                    ct_i
                )
                
                # Boundary check
                new_pos = np.clip(new_pos, self.lb, self.ub)
                updated_population[idx] = new_pos
                self.count[idx] = 0  # Reset counter
        
        return updated_population, updated_fitness
    
    def increment_nfes(self, count=1):
        """Increment number of function evaluations"""
        self.nfes += count

