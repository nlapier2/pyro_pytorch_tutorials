# implement the Adaline algorithm, presented in Chapter 2 of Raschka et al
import numpy as np

class AdalineGD:
    """
    Class implementing the Adaline algorithm, with classic Gradient Descent (GD).
    """


    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta  # learning rate
        self.n_iter = n_iter  # number of training iterations (epochs)
        self.random_state = 1


    def fit(self, X, y):
        """
        Loop to fit network weights.
        """
        rgen = np.random.RandomState(self.random_state)  # generator for random numbers
        self.w_ = rgen.normal(loc=0.0, scale=1.0, size=X.shape[1])  # weights initialized to random normal
        self.b_ = np.float_(0)
        self.losses_ = []  # tracks errors at each step

        for _ in range(self.n_iter):
            net_input = self.net_input(X)  # apply weights and bias
            output = self.activation(net_input)  # apply activation function to input
            errors = y - output
            # update weights and bias
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors ** 2).mean()  # mean squared error
            self.losses_.append(loss)
        return self
    

    def net_input(self, X):
        """
        Computes the input to the network, based on the trained weights,
            e.g. computes the decision boundary, which is used to make
            the predictions (net outputs).
        """
        return np.dot(X, self.w_) + self.b_


    def predict(self, X):
        """
        Make predictions, based on whether, given X, the net input is
        positive (in which case we predict class 1) or negative (class 0).
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    

    def activation(self, X):
        """Implements basic linear activation (no change)"""
        return X
    

# SGD version of AdaLine
class AdalineSGD:
    """
    Class implementing the Adaline algorithm, with Stochastic Gradient Descent (SGD).
    """


    def __init__(self, eta=0.01, n_iter=50, random_state=1, shuffle=True):
        self.eta = eta  # learning rate
        self.n_iter = n_iter  # number of training iterations (epochs)
        self.random_state = 1
        self.w_initialized = False 
        self.shuffle = shuffle  # shuffle data after each epoch to prevent cycles


    def fit(self, X, y):
        """
        Loop to fit network weights.
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []  # tracks errors at each step

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
                losses = []
                for xi, target in zip(X, y):
                    losses.append(self._update_weights(xi, target))
                avg_loss = np.mean(losses)
                self.losses_.append(avg_loss)
        return self
    

    def partial_fit(self, X, y):
        """ 
        fit training data without reinitializing weights.
        the idea is that we can use this for online learning, where we start with our
            previously trained weights and then update with one or more new data points.
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:  # if more than one data point (ravel flattens mat)
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self
    

    def _shuffle(self, X, y):
        """ Randomly shuffle data to avoid cycles """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    

    def _initialize_weights(self, m):
        """ Initialize weight vector and bias """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, scale = 1.0, size = m)
        self.b_ = np.float_(0.)
        self.w_initialized = True


    def _update_weights(self, xi, target):
        """ Apply Adaline learning rule to update weights """
        error = target - self.activation(self.net_input(xi))
        self.w_ += self.eta * 2.0 * error * xi 
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss


    def net_input(self, X):
        """
        Computes the input to the network, based on the trained weights,
            e.g. computes the decision boundary, which is used to make
            the predictions (net outputs).
        """
        return np.dot(X, self.w_) + self.b_


    def predict(self, X):
        """
        Make predictions, based on whether, given X, the net input is
        positive (in which case we predict class 1) or negative (class 0).
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    

    def activation(self, X):
        """Implements basic linear activation (no change)"""
        return X
    