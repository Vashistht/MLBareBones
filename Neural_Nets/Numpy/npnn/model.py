"""Neural Network model."""

from .modules import Module
from .optimizer import Optimizer
from tqdm import tqdm
import numpy as np


def categorical_cross_entropy(pred, labels, epsilon=1e-10):
    """Cross entropy loss function.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).
    epsilon : float
        Small constant to add to the log term of cross entropy to help
        with numerical stability.

    Returns
    -------
    float
        Cross entropy loss.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


def categorical_accuracy(pred, labels):
    """Accuracy statistic.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).

    Returns
    -------
    float
        Mean accuracy in this batch.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


class Sequential:
    """Sequential neural network model.

    Parameters
    ----------
    modules : Module[]
        List of modules; used to grab trainable weights.
    loss : Module
        Final output activation and loss function.
    optimizer : Optimizer
        Optimization policy to use during training.
    """

    def __init__(self, modules, loss=None, optimizer=None):

        for module in modules:
            assert(isinstance(module, Module))
        assert(isinstance(loss, Module))
        assert(isinstance(optimizer, Optimizer))

        self.modules = modules
        self.loss = loss

        self.params = []
        for module in modules:
            self.params += module.trainable_weights

        self.optimizer = optimizer
        self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        """Model forward pass.

        Parameters
        ----------
        X : np.array
            Input data

        Keyword Args
        ------------
        train : bool
            Indicates whether we are training or testing.

        Returns
        -------
        np.array
            Batch predictions; should have shape (batch, num_classes).
        """
        Batch = X.copy()
        for module in self.modules:
            Batch = module.forward(Batch, train)
        predictions = self.loss.forward(Batch)
        return predictions
    
        # raise NotImplementedError()

    def backward(self, y):
        """Model backwards pass.

        Parameters
        ----------
        y : np.array
            True labels.
        """
        y_copy = y.copy()
        y = self.loss.backward(y_copy) # to pass to backward
        for module in reversed(self.modules):
            y = module.backward(y) # new y to pass to next module
        # see if this handles the xk-1 case well 
        # raise NotImplementedError()

    def train(self, dataset):
        """Fit model on dataset for a single epoch.

        Parameters
        ----------
        X : np.array
            Input images
        dataset : Dataset
            Training dataset with batches already split.

        Notes
        -----
        You may find tqdm, which creates progress bars, to be helpful:

        Returns
        -------
        (float, float)
            [0] Mean train loss during this epoch.
            [1] Mean train accuracy during this epoch.
        """
        # raise NotImplementedError()
        train_loss, train_acc = [], []
        dataset = tqdm(dataset, desc="Training", leave=True)
        for X_batch, y_batch in dataset:
            # forrward pass and get loss
            predictions = self.forward(X_batch)
            
            loss = categorical_cross_entropy (predictions, y_batch)
            train_loss.append(loss)
            accuracy = categorical_accuracy(predictions, y_batch)
            train_acc.append(accuracy)
            
            # backward pass
            self.backward(y_batch)
            # update weights
            self.optimizer.apply_gradients(self.params)
            
        mean_loss, mean_acc = np.mean(train_loss), np.mean(train_acc)
        return mean_loss, mean_acc
        
        
    def test(self, dataset):
        """Compute test/validation loss for dataset.

        Parameters
        ----------
        dataset : Dataset
            Validation dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean test loss.
            [1] Test accuracy.
        """
        # raise NotImplementedError()
        
        test_loss,test_acc = [], []
        dataset = tqdm(dataset, desc="Training", leave=True)
        
        for X_batch, y_batch in dataset:
            predictions = self.forward(X_batch)
            loss = categorical_cross_entropy (predictions, y_batch)
            test_loss.append(loss)
            accuracy = categorical_accuracy(predictions, y_batch)
            test_acc.append(accuracy)
            
        # no back needed for test
            
        mean_loss, mean_acc = np.mean(test_loss), np.mean(test_acc)
        return mean_loss, mean_acc
        