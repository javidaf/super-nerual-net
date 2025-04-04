import numpy as np
from ml_p2.neural_network import BaseNeuralNetwork
from ml_p2.neural_network import Activation
from ml_p2.neural_network import Initializer
from ml_p2.neural_network.optimizer import SGD, Adam, RMSprop, AdaGrad


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(
        self,
        input_size=2,
        hidden_layers=[32],
        output_size=1,
        hidden_activation="sigmoid",
        output_activation="linear",
        initializer="normal",
        optimizer="adam",
        learning_rate=0.01,
        use_regularization=False,
        lambda_=0.01,
        classification_type="binary",
    ):
        """
        Initialize the neural network with given architecture.

        Parameters:
        - input_size: int, number of input features
        - hidden_layers: list of ints, number of neurons in each hidden layer
        - output_size: int, number of output neurons
        - hidden_activation: str, activation function for hidden layers
        - output_activation: str, activation function for output layer
        - initializer: str, initializer to use
        - optimizer: str, optimizer to use
        - learning_rate: float, learning rate for the optimizer

        """
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        self.classification_type = classification_type

        self.activations = []
        self.activation_derivatives = []
        for i in range(len(self.layers) - 1):
            if i < len(hidden_layers):
                act = getattr(Activation, hidden_activation)
                act_derivative = getattr(Activation, f"{hidden_activation}_derivative")
            else:
                act = getattr(Activation, output_activation)
                act_derivative = getattr(Activation, f"{output_activation}_derivative")
            self.activations.append(act)
            self.activation_derivatives.append(act_derivative)

        optimizer_map = {
            "sgd": SGD,
            "adam": Adam,
            "rmsprop": RMSprop,
            "adagrad": AdaGrad,
        }

        optimizer_class = optimizer_map.get(optimizer.lower(), Adam)
        self.optimizer = optimizer_class(learning_rate=learning_rate)
        self.initializer = getattr(Initializer, initializer)
        self.cost_history = []
        self.use_regularization = use_regularization
        self.lambda_ = lambda_

        self.params = {
            "hidden_activation": hidden_activation,
            "output_activation": output_activation,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "initializer": initializer,
            "hidden_layers": hidden_layers,
            "output_size": output_size,
            "input_size": input_size,
            "use_regularization": use_regularization,
            "lambda_": lambda_,
            "classification_type": classification_type,
        }

        self._initialize_parameters()

    def _initialize_parameters(self):
        for i in range(len(self.layers) - 1):
            weight = self.initializer(self.layers[i], self.layers[i + 1])
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        """
        Perform a forward pass through the network.

        Parameters:
        - X: numpy array of shape (n_samples, input_size)

        Returns:
        - activations: list of activations for each layer
        - zs: list of z vectors for each layer
        """
        activations = [X]
        zs = []
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], weight) + bias
            zs.append(z)
            a = self.activations[idx](z)
            activations.append(a)
        return activations, zs

    def backward(self, activations, zs, y):
        """
        Perform backward propagation to compute gradients of the loss function with respect to weights and biases.

        Parameters:
        - activations: list of activations from forward pass
        - zs: list of z vectors from forward pass
        - y: target values

        Returns:
        - grad_w: list of gradients for weights
        - grad_b: list of gradients for biases
        """
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        # output error
        # δ⁽L⁾ = (a⁽L⁾ - y) ⊙ σ'(z⁽L⁾)    shape: (m × nₗ)
        # For sigmoid activation function, σ'(z⁽L⁾) = σ(z⁽L⁾) * (1 - σ(z⁽L⁾))
        # Need to use activation in the input of the derivative function for sigmoid
        # For other activation functions, use zs
        if self.params["output_activation"] == "softmax":
            delta = activations[-1] - y  # Derivative of softmax + cross-entropy
        else:
            if self.activation_derivatives[-1].__name__ == "sigmoid_derivative":
                delta = (activations[-1] - y) * self.activation_derivatives[-1](
                    activations[-1]
                )
            else:
                delta = (activations[-1] - y) * self.activation_derivatives[-1](zs[-1])

        # ∇W⁽L⁾ = (a⁽L-1⁾)ᵀ · δ⁽L⁾        shape: (n_{L-1} × nₗ)
        grad_w[-1] = np.dot(activations[-2].T, delta)

        # ∇b⁽L⁾ = sum(δ⁽L⁾, axis=0)      shape: (1 × nₗ)
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        if self.use_regularization:
            grad_w[-1] += self.lambda_ * self.weights[-1]

        # backpropagate the error
        for l in range(2, len(self.layers)):
            # δ⁽l⁾ = (δ⁽l+1⁾ · (W⁽l⁾)ᵀ) ⊙ σ'(z⁽l⁾)    shape: (m × nₗ)
            if self.activation_derivatives[-l].__name__ == 'sigmoid_derivative':
                delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_derivatives[-l](activations[-l])
            else:
                delta = np.dot(delta, self.weights[-l + 1].T) * self.activation_derivatives[-l](zs[-l])

            # ∇W⁽l⁾ = (a⁽l-1⁾)ᵀ · δ⁽l⁾        shape: (n_{l-1} × nₗ)
            grad_w[-l] = np.dot(activations[-l-1].T, delta)

            if self.use_regularization:
                grad_w[-l] += self.lambda_ * self.weights[-l]

            # ∇b⁽l⁾ = sum(δ⁽l⁾, axis=0)      shape: (1 × nₗ)
            grad_b[-l] = np.sum(delta, axis=0, keepdims=True)

        return grad_w, grad_b

    def compute_loss(self, y_pred, y):
        """Compute appropriate loss based on classification type"""
        if self.classification_type == "binary":
            return self.compute_binary_cross_entropy(y_pred, y)
        elif self.classification_type == "multiclass":
            return self.compute_categorical_cross_entropy(y_pred, y)
        else:  # regression
            return self.compute_mse(y_pred, y)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def train(self, X, y, epochs, batch_size=32):
        n_samples = X.shape[0]
        self.optimizer.initialize(self.weights, self.biases)
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                activations, zs = self.forward(X_batch)
                grad_w, grad_b = self.backward(activations, zs, y_batch)
                
                # Update weights and biases using the chosen optimizer
                self.weights, self.biases = self.optimizer.update(
                    self.weights, self.biases, grad_w, grad_b
                )
            
            # Compute metrics for the whole dataset
            predictions, _ = self.forward(X)
            loss = self.compute_loss(predictions[-1], y)
            r2 = self.score(X, y)
            self.cost_history.append(loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, R2: {r2:.4f}")

    def predict(self, X):
        """Make predictions with the trained neural network."""
        activations, _ = self.forward(X)
        return activations[-1]

    ######------------Classification-----
    def compute_categorical_cross_entropy(self, y_pred, y_true):
        """Compute categorical cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # If labels are one-hot encoded
        if len(y_true.shape) == 2:
            loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        else:
            # If labels are integer encoded
            n_samples = y_true.shape[0]
            loss = -np.sum(np.log(y_pred[range(n_samples), y_true])) / n_samples

        if self.use_regularization:
            l2_reg = 0
            for w in self.weights:
                l2_reg += np.sum(w**2)
            return loss + 0.5 * self.lambda_ * l2_reg
        return loss

    def compute_binary_cross_entropy(self, y_pred, y_true):
        """Compute binary cross-entropy loss with optional L2 regularization"""
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(
            y_pred, epsilon, 1 - epsilon
        )  # Clip predictions to avoid numerical instability

        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        if self.use_regularization:
            l2_reg = 0
            for w in self.weights:
                l2_reg += np.sum(w**2)
            return loss + 0.5 * self.lambda_ * l2_reg
        return loss

    def predict_proba(self, X):
        """Predict class probabilities"""
        activations, _ = self.forward(X)
        return activations[-1]

    def predict_classes(self, X):
        """Predict class labels"""
        if self.classification_type == "binary":
            probas = self.predict_proba(X)
            return (probas > 0.5).astype(int)
        else:
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)

    def accuracy_score(self, X, y):
        """Compute accuracy score"""
        y_pred = self.predict_classes(X)
        if len(y.shape) == 2:  # If one-hot encoded
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y
        return np.mean(y_pred == y_true)

    def train_classifier(self, X, y, epochs, batch_size=32):
        """Train the network for classification"""
        n_samples = X.shape[0]
        self.optimizer.initialize(self.weights, self.biases)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                activations, zs = self.forward(X_batch)
                grad_w, grad_b = self.backward(activations, zs, y_batch)

                self.weights, self.biases = self.optimizer.update(
                    self.weights, self.biases, grad_w, grad_b
                )

            predictions, _ = self.forward(X)
            if self.classification_type == "binary":
                loss = self.compute_binary_cross_entropy(predictions[-1], y)
            else:
                loss = self.compute_categorical_cross_entropy(predictions[-1], y)
            accuracy = self.accuracy_score(X, y)
            self.cost_history.append(loss)

            # if (epoch + 1) % 10 == 0 or epoch == 0:
            #     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
