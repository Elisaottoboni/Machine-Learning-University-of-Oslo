import numpy as np
import pandas as pd
import autograd.numpy as np
from autograd import grad
from sklearn.metrics import mean_squared_error

class GradientDescendAG:
    def __init__(self, optimizer="gd", learning_rate=0.001, max_epochs=100, batch_size=20,
                learning_rate_decay=0.9, patience=20, delta_momentum=0.3, lmb=0.001,
                tol=1e-8, delta=1e-8, rho=0.9, beta1=0.9, beta2=0.99, momentum=True,
                learning_rate_decay_flag=False, Ridge=False, method=None, change=0.0):
        # Parameter configuration
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = int(batch_size)
        self.learning_rate_decay = learning_rate_decay
        self.patience = patience
        self.delta_momentum = delta_momentum if momentum else 0
        self.lmb = lmb if Ridge else 0
        self.tol = tol
        self.delta = delta
        self.rho = rho
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate_decay_flag = learning_rate_decay_flag
        self.method = method
        
        # State variables for optimization
        self.theta = 0.0
        self.first_moment = 0.0
        self.second_moment = 0.0
        self.iter = 0
        self.change = change
        self.gradient_squared = 0.0  # Cumulative gradient for Adagrad and RMSprop

    def compute_hessian_eig_max(self, X):
        """Calculate the maximum eigenvalue of the Hessian, useful for defining an optimal learning rate."""
        H = (2.0 / len(X)) * np.matmul(X.T, X)
        if self.lmb:
            H += 2 * self.lmb * np.eye(X.shape[1])
        return 1.0 / np.max(np.linalg.eigvals(H))
    
    def compute_gradient(self, X, y, thetas):
        """Compute the gradient based on the difference between predictions and targets, with regularization."""
        residuals = np.dot(X, thetas) - y
        gradient = 2.0 * np.dot(X.T, residuals) / len(X) + 2 * self.lmb * thetas
        return gradient
    
    def cost_function(self, X, y, beta):
        """Compute the cost function, adding an L2 norm penalty if Ridge regularization is applied."""
        residuals = np.dot(X, beta) - y
        cost = np.sum(residuals ** 2) / len(X) + self.lmb * (np.sqrt(np.sum(beta ** 2)))**2
        return cost
    
    def _learning_schedule(self, t):
        """Generate a variable learning rate for learning rate decay."""
        return self.learning_rate / (1 + self.learning_rate_decay * t)
    
    def gd_step(self, X, y, thetas):
        """Gradient descent step with optional momentum."""
        gradient = self.compute_gradient(X, y, thetas)
        change = self.learning_rate * gradient + self.delta_momentum * self.change
        thetas -= change
        self.change = change
        return thetas

    def RMSprop_step(self, X, y, thetas):
        """RMSprop step, which updates the gradient with a moving average to stabilize updates."""
        gradients = self.compute_gradient(X, y, thetas)
        self.gradient_squared = self.rho * self.gradient_squared + (1 - self.rho) * gradients ** 2
        thetas -= self.learning_rate * gradients / (self.delta + np.sqrt(self.gradient_squared))
        return thetas
    
    def ADAM_step(self, X, y, thetas):
        """ADAM optimization step using momentum for first and second moments."""
        gradients = self.compute_gradient(X, y, thetas)
        self.first_moment = self.beta1 * self.first_moment + (1 - self.beta1) * gradients
        self.second_moment = self.beta2 * self.second_moment + (1 - self.beta2) * gradients ** 2
        first_unbiased = self.first_moment / (1 - self.beta1 ** self.iter)
        second_unbiased = self.second_moment / (1 - self.beta2 ** self.iter)
        thetas -= self.learning_rate * first_unbiased / (np.sqrt(second_unbiased) + self.delta)
        return thetas
    
    def Adagrad_GD_step(self, X, y, thetas):
        """Adagrad step, adjusting the learning rate based on cumulative gradients."""
        gradients = self.compute_gradient(X, y, thetas)
        self.gradient_squared += gradients ** 2
        thetas -= self.learning_rate * gradients / (self.delta + np.sqrt(self.gradient_squared))
        return thetas
    
    # Optimization methods
    def gradient_descent(self, X_train, y_train, X_val, y_val):
        """Implementation of standard Gradient Descent with optional momentum and learning rate decay."""
        
        thetas = np.random.randn(X_train.shape[1], 1)
        patience_counter = 0
        best_val_error = float('inf')
        best_thetas = np.copy(thetas)
        
        for epoch in range(self.max_epochs):
            thetas = self.gd_step(X_train, y_train, thetas)
            if self.learning_rate_decay_flag:
                val_loss = mean_squared_error(y_val, np.dot(X_val, thetas))
                if val_loss < best_val_error - self.tol:
                    best_val_error = val_loss
                    best_thetas = thetas
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > self.patience:
                        self.learning_rate *= self.learning_rate_decay
                        patience_counter = 0
        return best_thetas

    def stochastic_gradient_descent(self, X_train, y_train, X_val, y_val):
        """Implementation of Stochastic Gradient Descent with optional momentum and batch processing."""
        
        num_samples = len(X_train)
        thetas = np.random.randn(X_train.shape[1],1)
        patience_counter = 0
        best_val_error = float('inf')
        best_thetas = np.copy(thetas)
        
        # Select optimization method based on user input
        if self.method == 'sgd':
            method_ = self.gd_step
        elif self.method == 'RMSprop':
            method_ = self.RMSprop_step
        elif self.method == 'ADAM':
            self.m = np.zeros((X_train.shape[1], 1))
            self.v = np.zeros((X_train.shape[1], 1))
            self.t = 0
            method_ = self.ADAM_step
        elif self.method == 'Adagrad_GD':
            method_ = self.Adagrad_GD_step
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
        
        # Loop over epochs and batch the dataset
        for epoch in range(self.max_epochs):
            index = np.random.permutation(num_samples)
            self.iter += 1
            for i in range(0, num_samples, self.batch_size):
                random_index = index[i:i + self.batch_size]
                batch_X = X_train[random_index]
                batch_y = y_train[random_index]
                thetas = method_(batch_X, batch_y, thetas)
        
        # Update learning rate if decay is enabled
        if self.learning_rate_decay_flag:
            val_loss = mean_squared_error(y_val, np.dot(X_val, thetas))
            if val_loss < best_val_error - self.tol:
                best_val_error = val_loss
                best_thetas = thetas
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > self.patience and self.learning_rate_decay_flag:
                    self.learning_rate *= self.learning_rate_decay
        return best_thetas
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Main training method."""
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        
        if self.optimizer == "gd":
            self.thetas = self.gradient_descent(X_train, y_train, X_val, y_val)
            return self.thetas
        elif self.optimizer == "sgd":
            self.thetas = self.stochastic_gradient_descent(X_train, y_train, X_val, y_val)
            return self.thetas
        else:
            raise ValueError("Unsupported optimizer. Use 'gd' or 'sgd'.")
        
    def predict(self, X):
        return np.dot(X, self.thetas).reshape(-1, 1)