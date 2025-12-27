"""
DE-NPS Strategy Adapter for Non-DE Algorithms
Adapts DE-NPS strategies (parameter adaptation, logarithmic spiral perturbation, 
and diversity enhancement) to work with various optimization algorithms.
"""

import numpy as np
try:
    from scipy.linalg import eigh
except ImportError:
    # Fallback to numpy if scipy is not available
    from numpy.linalg import eigh


class DENPSAdapter:
    """Adapter for applying DE-NPS strategies to non-DE algorithms"""
    
    def __init__(self, dim, pop_size, max_generations, bounds):
        self.dim = dim
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.bounds = np.array(bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        
        self.nfes = 0
        self.nfes_max = max_generations * pop_size
        
        # Parameter adaptation memory (for algorithm-specific parameters)
        self.H = 4
        self.mu_param = np.ones(self.H) * 0.5  # Generic parameter mean
        self.memory_index = 0
        self.S_param = []  # Successful parameters
        
        # Stagnation detection
        self.count = np.zeros(pop_size)
        self.count_threshold = 2 * dim
        
        # Perturbation parameters
        self.tau = 0.1
        self.b = 1.0
        
        # Stage threshold
        self.l = int(self.nfes_max * 0.2)
    
    def tripuls(self, t, width=1.0):
        """Triangular pulse function"""
        t = np.abs(t)
        if t >= width:
            return 0.0
        return 1.0 - t / width
    
    def get_adaptive_parameter(self, base_value=0.5):
        """
        Bi-stage parameter adaptation
        Returns an adaptive parameter value based on current stage
        """
        r = np.random.randint(0, self.H)
        mu_r = self.mu_param[r]
        
        if self.nfes < self.l:
            # Stage 1: Exploration - use tripuls
            rand_val = np.random.rand()
            rand_1 = np.random.rand()
            param = mu_r + 0.1 * rand_val * self.tripuls(rand_1, mu_r)
            param = np.clip(param, 0.1, 1.0)
        else:
            # Stage 2: Exploitation - use Cauchy distribution
            param = np.random.standard_cauchy() * 0.1 + mu_r
            param = np.clip(param, 0.1, 1.0)
        
        return param
    
    def logarithmic_spiral_perturbation(self, X_i, X_new):
        """
        Apply logarithmic spiral perturbation to a new solution
        """
        # Calculate distance
        Dis = X_i - X_new
        
        # Parameters for logarithmic spiral
        a = -1.0 + self.nfes / self.nfes_max
        l = (a - 1.0) * np.random.rand() + 1.0
        
        # Apply perturbation with probability tau
        if np.random.rand() < self.tau:
            # Logarithmic spiral perturbation
            X_perturbed = Dis * np.exp(l * self.b) * np.cos(l * 2 * np.pi) + X_i
            # Blend with original
            alpha = np.random.rand()
            X_new = alpha * X_new + (1 - alpha) * X_perturbed
        
        return X_new
    
    def compute_covariance_matrix(self, population):
        """Compute covariance matrix of the population"""
        mean_pop = np.mean(population, axis=0)
        centered_pop = population - mean_pop
        
        C = np.cov(centered_pop.T)
        
        if C.ndim == 0:
            C = np.array([[C]])
        elif C.ndim == 1:
            C = np.diag(C)
        
        return C, mean_pop
    
    def pca_reduction(self, population, n_components=None):
        """Perform PCA reduction on population"""
        if n_components is None:
            n_components = max(1, int(self.pop_size * 0.2))
        
        C, mean_pop = self.compute_covariance_matrix(population)
        
        try:
            eigenvalues, eigenvectors = eigh(C)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            if n_components > len(eigenvalues):
                n_components = len(eigenvalues)
            
            selected_eigenvectors = eigenvectors[:, :n_components]
            centered_pop = population - mean_pop
            X_pca = np.dot(centered_pop, selected_eigenvectors)
            
            return X_pca, mean_pop, selected_eigenvectors
        except:
            return population, mean_pop, np.eye(self.dim)
    
    def update_stagnant_individuals(self, population, fitness, best_solution, best_fitness):
        """
        Detect and update stagnant individuals using covariance matrix and PCA
        """
        updated_population = population.copy()
        updated_fitness = fitness.copy()
        
        # Update stagnation counters
        for i in range(len(population)):
            if fitness[i] >= best_fitness:  # No improvement
                self.count[i] += 1
            else:
                self.count[i] = 0
        
        # Find stagnant individuals
        stagnant_indices = np.where(self.count > self.count_threshold)[0]
        
        if len(stagnant_indices) > 0:
            # Perform PCA on population
            X_pca, mean_pop, eigenvectors = self.pca_reduction(
                population, 
                n_components=max(1, int(len(population) * 0.2))
            )
            
            # Update each stagnant individual
            for idx in stagnant_indices:
                if len(X_pca) >= 2:
                    pca_indices = np.random.choice(len(X_pca), 2, replace=False)
                    X_pca1 = X_pca[pca_indices[0]]
                    X_pca2 = X_pca[pca_indices[1]]
                else:
                    X_pca1 = X_pca[0] if len(X_pca) > 0 else np.zeros(self.dim)
                    X_pca2 = X_pca[0] if len(X_pca) > 0 else np.zeros(self.dim)
                
                # Project back to original space
                if eigenvectors.shape[1] > 0:
                    X_pca1_full = mean_pop + np.dot(X_pca1, eigenvectors.T)
                    X_pca2_full = mean_pop + np.dot(X_pca2, eigenvectors.T)
                else:
                    X_pca1_full = mean_pop
                    X_pca2_full = mean_pop
                
                # Apply one of two mutation equations
                if np.random.rand() < 0.5:
                    # Equation 11: X_i = X_pca1 - rand * (X_pca1 - X_best)
                    new_pos = X_pca1_full - np.random.rand() * (X_pca1_full - best_solution)
                else:
                    # Equation 12: X_i = X_best + rand * (X_pca1 - X_pca2)
                    new_pos = best_solution + np.random.rand() * (X_pca1_full - X_pca2_full)
                
                # Boundary check
                new_pos = np.clip(new_pos, self.lb, self.ub)
                updated_population[idx] = new_pos
                self.count[idx] = 0  # Reset counter
        
        return updated_population, updated_fitness
    
    def increment_nfes(self, count=1):
        """Increment number of function evaluations"""
        self.nfes += count

