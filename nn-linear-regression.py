import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self):
        self.weights = None
        self.bias = None

    def train(self, X_train, y_train, learning_rate=0.01, epochs=100):
        # Initialize weights and bias
        np.random.seed(1)
        self.weights = np.random.randn(1)
        self.bias = np.random.randn(1)

        for epoch in range(epochs):
            total_loss = self._train_one_epoch(X_train, y_train, learning_rate)

            # Print the total loss for this epoch
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Total Loss: {total_loss}')

    def _train_one_epoch(self, X_train, y_train, learning_rate):
        total_loss = 0

        for i in range(len(X_train)):
            # Forward pass for a single data point
            prediction = self.predict(X_train[i])

            # Compute the mean squared error for this data point
            loss = (prediction - y_train[i]) ** 2
            total_loss += loss

            # Backward pass (gradient descent) for a single data point
            grad_weights, grad_bias = self._backpropagate(X_train[i], y_train[i], prediction)

            # Update weights and bias for a single data point
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias

        return total_loss

    def _backpropagate(self, x, y_true, y_pred):
        # Chain rule for gradients
        grad_loss = 2 * (y_pred - y_true)
        grad_weights = grad_loss * x
        grad_bias = grad_loss

        return grad_weights, grad_bias

    def predict(self, x):
        return x * self.weights + self.bias

# Generate some random data for training
np.random.seed(0)
X_train = np.random.rand(100, 1)
y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1)

# Create and train the linear regression model
linear_model = LinearRegressionModel()
linear_model.train(X_train, y_train)

# Test the model on new data
X_test = np.array([[0.2], [0.5], [0.8]])
predictions = linear_model.predict(X_test)

# Print the predictions
print("Predictions:")
print(predictions)

# Plot the training data and the learned line
plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, linear_model.predict(X_train), color='red', label='Learned Line')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()
