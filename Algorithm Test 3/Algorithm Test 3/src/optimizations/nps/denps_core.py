"""
DE-NPS Core Strategies Module
Implements three key strategies from the DE-NPS paper:
1. Bi-stage parameter adaptation scheme
2. Logarithmic spiral perturbation strategy
3. Population diversity enhancement mechanism based on covariance matrix
"""

import numpy as np
from scipy.stats import cauchy
from scipy.linalg import eigh


class DENPSStrategies:
    """Core strategies from DE-NPS algorithm"""
    
    def __init__(self, dim, pop_size, max_generations):
        self.dim = dim
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.nfes = 0
        self.nfes_max = max_generations * pop_size
        
        # Parameter adaptation memory
        self.H = 4  # Memory size
        self.mu_F = np.ones(self.H) * 0.5  # Mean F values
        self.mu_CR = np.ones(self.H) * 0.5  # Mean CR values
        self.memory_index = 0
        
        # Successful parameter storage
        self.S_F = []
        self.S_CR = []
        
        # Stagnation detection
        self.count = np.zeros(pop_size)  # Stagnation counter for each individual
        self.count_threshold = 2 * dim  # Threshold for stagnation
        
        # Perturbation parameters
        self.tau = 0.1  # Crossover strategy selection probability
        self.b = 1.0  # Logarithmic spiral shape parameter
        
        # Stage threshold
        self.l = int(self.nfes_max * 0.2)  # 20% of max evaluations
    
    def tripuls(self, t, width=1.0):
        """
        Triangular pulse function
        Returns a triangular pulse centered at 0 with specified width
        """
        t = np.abs(t)
        if t >= width:
            return 0.0
        return 1.0 - t / width
    
    def get_parameters(self):
        """
        Bi-stage parameter adaptation scheme
        Returns F and CR values based on current stage
        """
        # Select random index from memory
        r = np.random.randint(0, self.H)
        mu_F_r = self.mu_F[r]
        mu_CR_r = self.mu_CR[r]
        
        # Stage 1: Early stage (exploration) - use tripuls
        if self.nfes < self.l:
            # F generation using tripuls
            rand_val = np.random.rand()
            rand_1 = np.random.rand()
            F = mu_F_r + 0.1 * rand_val * self.tripuls(rand_1, mu_F_r)
            F = np.clip(F, 0.1, 1.0)
        else:
            # Stage 2: Later stage (exploitation) - use Cauchy distribution
            F = np.random.standard_cauchy() * 0.1 + mu_F_r
            F = np.clip(F, 0.1, 1.0)
        
        # CR generation
        if mu_CR_r == 0:
            CR = 0
        else:
            CR = np.random.normal(mu_CR_r, 0.1)
            CR = np.clip(CR, 0.0, 1.0)
        
        return F, CR
    
    def logarithmic_spiral_perturbation(self, X_i, U_i, CR, j_rand):
        """
        Logarithmic spiral perturbation strategy
        Applied during crossover operation
        """
        U_new = U_i.copy()
        
        # Calculate distance
        Dis = X_i - U_i
        
        # Parameters for logarithmic spiral
        a = -1.0 + self.nfes / self.nfes_max  # Linearly decreases from -1 to 0
        l = (a - 1.0) * np.random.rand() + 1.0  # Random number in [a, 1]
        
        # Apply perturbation to each dimension
        for j in range(self.dim):
            if j == j_rand or np.random.rand() < CR:
                # Standard crossover
                U_new[j] = U_i[j]
            else:
                # Apply logarithmic spiral perturbation
                if np.random.rand() > self.tau:
                    # Standard crossover
                    U_new[j] = X_i[j]
                else:
                    # Logarithmic spiral perturbation
                    U_new[j] = Dis[j] * np.exp(l * self.b) * np.cos(l * 2 * np.pi) + X_i[j]
        
        return U_new
    
    def update_parameters(self, successful_F, successful_CR):
        """
        Update parameter memory based on successful parameters
        """
        if len(successful_F) > 0:
            # Weight calculation based on fitness improvement
            weights = np.array(successful_F)  # Simplified: use F values as weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                
                # Weighted Lehmer mean for F
                mean_F = np.sum(weights * np.array(successful_F)**2) / np.sum(weights * np.array(successful_F))
                self.mu_F[self.memory_index] = mean_F
            else:
                self.mu_F[self.memory_index] = self.mu_F[self.memory_index]
        
        if len(successful_CR) > 0:
            # Weighted mean for CR
            weights = np.array(successful_CR)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                mean_CR = np.sum(weights * np.array(successful_CR))
                self.mu_CR[self.memory_index] = mean_CR
            else:
                self.mu_CR[self.memory_index] = self.mu_CR[self.memory_index]
        
        # Update memory index
        self.memory_index = (self.memory_index + 1) % self.H
        
        # Clear successful parameters
        self.S_F = []
        self.S_CR = []
    
    def compute_covariance_matrix(self, population):
        """
        Compute covariance matrix of the population
        """
        # Center the population
        mean_pop = np.mean(population, axis=0)
        centered_pop = population - mean_pop
        
        # Compute covariance matrix
        C = np.cov(centered_pop.T)
        
        # Ensure it's a proper matrix
        if C.ndim == 0:
            C = np.array([[C]])
        elif C.ndim == 1:
            C = np.diag(C)
        
        return C, mean_pop
    
    def pca_reduction(self, population, n_components=None):
        """
        Perform PCA reduction on population
        """
        if n_components is None:
            n_components = max(1, int(self.pop_size * 0.2))
        
        C, mean_pop = self.compute_covariance_matrix(population)
        
        # Eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = eigh(C)
            # Sort by eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Select top n_components
            if n_components > len(eigenvalues):
                n_components = len(eigenvalues)
            
            selected_eigenvectors = eigenvectors[:, :n_components]
            
            # Project population to reduced space
            centered_pop = population - mean_pop
            X_pca = np.dot(centered_pop, selected_eigenvectors)
            
            return X_pca, mean_pop, selected_eigenvectors
        except:
            # Fallback: return original population
            return population, mean_pop, np.eye(self.dim)
    
    def update_stagnant_individuals(self, population, fitness, best_solution, best_fitness):
        """
        Detect and update stagnant individuals using covariance matrix and PCA
        """
        updated_population = population.copy()
        updated_fitness = fitness.copy()
        
        # Update stagnation counters
        for i in range(self.pop_size):
            if fitness[i] >= best_fitness:  # No improvement
                self.count[i] += 1
            else:
                self.count[i] = 0
        
        # Find stagnant individuals
        stagnant_indices = np.where(self.count > self.count_threshold)[0]
        
        if len(stagnant_indices) > 0:
            # Perform PCA on population
            X_pca, mean_pop, eigenvectors = self.pca_reduction(population, n_components=max(1, int(self.pop_size * 0.2)))
            
            # Update each stagnant individual
            for idx in stagnant_indices:
                # Select two random individuals from PCA space
                if len(X_pca) >= 2:
                    pca_indices = np.random.choice(len(X_pca), 2, replace=False)
                    X_pca1 = X_pca[pca_indices[0]]
                    X_pca2 = X_pca[pca_indices[1]]
                else:
                    # Fallback
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
                lb = self.bounds[:, 0] if hasattr(self, 'bounds') else -100
                ub = self.bounds[:, 1] if hasattr(self, 'bounds') else 100
                new_pos = np.clip(new_pos, lb, ub)
                
                updated_population[idx] = new_pos
                self.count[idx] = 0  # Reset counter
        
        return updated_population, updated_fitness
    
    def set_bounds(self, bounds):
        """Set bounds for boundary checking"""
        self.bounds = np.array(bounds)

