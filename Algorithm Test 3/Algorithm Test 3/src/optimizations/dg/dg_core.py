import numpy as np

class AdamOptimizer:
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        
    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class SimpleMLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        # Architecture: Input -> Hidden1 -> ReLU -> Hidden2 -> ReLU -> Output
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        # He/Xavier Initialization
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
        self.optimizer = AdamOptimizer(self.weights + self.biases, learning_rate)
        
    def forward(self, X, training=False):
        self.activations = [X]
        self.pre_activations = []
        
        input_data = X
        for i in range(len(self.weights)):
            z = np.dot(input_data, self.weights[i]) + self.biases[i]
            self.pre_activations.append(z)
            
            if i < len(self.weights) - 1:
                a = np.maximum(0, z) # ReLU
            else:
                a = z # Linear output
                
            self.activations.append(a)
            input_data = a
            
        return input_data
        
    def train_step(self, X, y):
        # Forward
        output = self.forward(X, training=True)
        
        # MSE Loss gradients
        batch_size = X.shape[0]
        grad_output = 2.0 * (output - y) / batch_size
        
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        # Backprop
        delta = grad_output
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient wrt weights and biases
            grad_weights[i] = np.dot(self.activations[i].T, delta)
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                # Gradient wrt input of this layer (output of prev layer)
                delta = np.dot(delta, self.weights[i].T)
                # ReLU derivative
                delta[self.pre_activations[i-1] <= 0] = 0
                
        # Update
        self.optimizer.step(grad_weights + grad_biases)
        
        loss = np.mean((output - y) ** 2)
        return loss

class InsightsGuider:
    def __init__(self, dim, history_size=200):
        self.dim = dim
        self.history_size = history_size
        
        # Buffer
        self.data_X = []
        self.data_y = []
        
        # Model
        # Simple architecture relative to dim
        h_dim = max(20, dim * 2)
        self.model = SimpleMLP(dim, [h_dim, h_dim], dim, learning_rate=0.005)
        self.is_ready = False
        self.train_counter = 0
        
    def store(self, x_curr, x_improved):
        if len(self.data_X) >= self.history_size:
            # Random replace or FIFO? FIFO is simpler
            self.data_X.pop(0)
            self.data_y.pop(0)
            
        self.data_X.append(x_curr)
        self.data_y.append(x_improved)
        
    def train(self, batch_size=32, epochs=5):
        if len(self.data_X) < batch_size:
            return
            
        X = np.array(self.data_X)
        y = np.array(self.data_y)
        
        # Simple shuffling
        indices = np.arange(len(X))
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                self.model.train_step(X[batch_idx], y[batch_idx])
                
        self.is_ready = True
        
    def predict(self, x):
        if not self.is_ready:
            return x
        
        # Ensure 2D
        input_x = x.reshape(1, -1) if x.ndim == 1 else x
        pred = self.model.forward(input_x)
        return pred.flatten() if x.ndim == 1 else pred

