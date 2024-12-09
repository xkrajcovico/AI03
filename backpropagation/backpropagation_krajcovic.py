import numpy as np
import matplotlib.pyplot as plt
import csv

# Define abstract base layer
class AbstractLayer:
    def forward(self, input_block):
        raise NotImplementedError("Forward pass not implemented.")
    def backward(self, upstream_grad):
        raise NotImplementedError("Backward pass not implemented.")
    def update(self, step_size, inertia=0):
        pass

# Fully-connected layer
class FullyConnected(AbstractLayer):
    def __init__(self, dim_in, dim_out):
        range_bound = np.sqrt(6.0 / (dim_in + dim_out))
        # Weight initialization
        self.W = np.random.uniform(-range_bound, range_bound, (dim_in, dim_out))
        self.b = np.zeros((1, dim_out))
        
        self.gradW = np.zeros_like(self.W)
        self.gradB = np.zeros_like(self.b)
        self.velW = np.zeros_like(self.W)
        self.velB = np.zeros_like(self.b)

    def forward(self, input_block):
        self.inp_cache = input_block
        return np.dot(input_block, self.W) + self.b

    def backward(self, upstream_grad):
        self.gradW = np.dot(self.inp_cache.T, upstream_grad)
        self.gradB = np.sum(upstream_grad, axis=0, keepdims=True)
        return np.dot(upstream_grad, self.W.T)

    def update(self, step_size, inertia=0):
        self.velW = inertia * self.velW - step_size * self.gradW
        self.velB = inertia * self.velB - step_size * self.gradB
        self.W += self.velW
        self.b += self.velB

# ----------------------------------------------------------------------
# Activation functions
# ----------------------------------------------------------------------
class ActivationBase(AbstractLayer):
    def forward(self, input_block):
        raise NotImplementedError("Forward not implemented.")
    def backward(self, upstream_grad):
        raise NotImplementedError("Backward not implemented.")

class Logistic(ActivationBase): # Sigmoid
    def forward(self, input_block):
        self.out_cache = 1.0 / (1.0 + np.exp(-input_block))
        return self.out_cache
    def backward(self, upstream_grad):
        return upstream_grad * self.out_cache * (1.0 - self.out_cache)

class HyperbolicTangent(ActivationBase): # Tanh
    def forward(self, input_block):
        self.out_cache = np.tanh(input_block)
        return self.out_cache
    def backward(self, upstream_grad):
        return upstream_grad * (1.0 - self.out_cache**2)

class RectifiedLinear(ActivationBase): # ReLU
    def forward(self, input_block):
        self.storage = input_block
        return np.maximum(0.0, input_block)
    def backward(self, upstream_grad):
        temp_grad = upstream_grad.copy()
        temp_grad[self.storage <= 0] = 0
        return temp_grad

# Mean Squared Error loss
class MeanSquaredError:
    def forward(self, preds, targets):
        self.stored_preds = preds
        self.stored_targs = targets
        return np.mean((preds - targets)**2)
    def backward(self):
        return 2.0 * (self.stored_preds - self.stored_targs) / self.stored_targs.size

# Model container
class SequentialModel:
    def __init__(self):
        self.seq_layers = []
    def add(self, layer):
        self.seq_layers.append(layer)
    def predict(self, X):
        temp = X
        for ly in self.seq_layers:
            temp = ly.forward(temp)
        return temp
    def backward_pass(self, grad_final):
        g = grad_final
        for ly in reversed(self.seq_layers):
            g = ly.backward(g)
    def update_params(self, alpha, mom=0):
        for ly in self.seq_layers:
            ly.update(alpha, mom)

def fit_model(model, loss_fn, training_inp, training_out, n_epochs, lr, mom=0):
    record_loss = []
    
    # We'll store log lines in a list, then write them out to CSV at the end
    log_lines = []
    log_lines.append(["Epoch", "Loss"])

    for ep in range(n_epochs):
        # Forward propagation
        guess = model.predict(training_inp)
        lss = loss_fn.forward(guess, training_out)
        record_loss.append(lss)

        # Backward propagation
        grad_lss = loss_fn.backward()
        model.backward_pass(grad_lss)

        # Parameter update
        model.update_params(lr, mom)

        if (ep == 0) or ((ep+1) % 50 == 0):
            print("Epoch {}/{} | Loss: {:.4f}".format(ep+1, n_epochs, lss))
        # Always log the result
        log_lines.append([ep+1, lss])

    # After training, write logs to results.csv
    with open("results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(log_lines)

    return record_loss

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # XOR data
    inputs = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ], dtype=float)

    targets = np.array([ #if you change theese values, you can train for AND or OR instead of XOR
        [1],
        [0],
        [0],
        [0]
    ], dtype=float)

    np.random.seed(42)
    net = SequentialModel()
    net.add(FullyConnected(2, 4))
    net.add(RectifiedLinear())  # ReLU for hidden layer
    net.add(FullyConnected(4, 4))
    net.add(HyperbolicTangent())  # Tanh for second hidden layer
    net.add(FullyConnected(4, 1))
    net.add(Logistic())  # Sigmoid for output

    # Define the loss
    mse_loss = MeanSquaredError()   

    # Training settings
    total_epochs = 500
    learning_rate = 0.1
    momen_val = 0.9

    # Train
    history = fit_model(net, mse_loss, inputs, targets, total_epochs, learning_rate, momen_val)

    # Plotting the training curve
    plt.plot(history)
    plt.title("Loss Over Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig('backpropagation_result.png')
    plt.show()

    # Predictions
    final_preds = net.predict(inputs)
    predict_binary = (final_preds > 0.5).astype(int)
    print("Predicted:")
    print(predict_binary.astype(int).ravel())
    print("Actual:")
    print(targets.astype(int).ravel())
