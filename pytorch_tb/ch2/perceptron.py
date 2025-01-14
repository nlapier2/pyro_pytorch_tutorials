# implement the basic Rosenblatt Perceptron, presented in Chapter 2 of Raschka et al
import numpy as np

class Perceptron:
    """
    Class implementing the basic Rosenblatt perceptron.
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
        self.errors = []  # tracks errors at each step

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors.append(errors)
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
        return np.where(self.net_input(X) >= 0.0, 1, 0)