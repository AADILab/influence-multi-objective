from typing import Union, List
import numpy as np

# Neural network class for evaluation
class NeuralNetwork:
    def __init__(self, num_inputs: int, num_hidden: Union[int, List[int]], num_outputs: int) -> None:
        if type(num_hidden) == int:
            num_hidden = [num_hidden]
        self.num_inputs, self.num_hidden, self.num_outputs = num_inputs, num_hidden, num_outputs
        # Number of nodes in each layer
        self.shape = tuple([self.num_inputs] + self.num_hidden + [self.num_outputs])
        # Number of layers
        self.num_layers = len(self.shape) - 1
        # Initialize weights with zeros
        self.weights = self.initWeights()
        # Store shape and size for later
        self.num_weights = self.calculateNumWeights()

    def shape(self):
        return (self.num_inputs, self.num_hidden, self.num_outputs)

    def initWeights(self) -> List[np.ndarray]:
        """Creates the numpy arrays for holding weights. Initialized to zeros """
        weights = []
        for num_inputs, num_outputs in zip(self.shape[:-1], self.shape[1:]):
            weights.append(np.zeros(shape=(num_inputs+1, num_outputs)))
        return weights

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Input layer is not activated.
        # We treat it as an activated layer so that we don't activate it.
        # (you wouldn't activate an already activated layer)
        a = X
        # Feed forward through each layer of hidden units and the last layer of output units
        for layer_ind in range(self.num_layers):
            # Add bias term
            b = np.hstack((a, [1]))
            # Feedforward through the weights w. summations
            f = b.dot(self.weights[layer_ind])
            # Activate the summations
            a = self.activation(f)
        return a

    def setWeights(self, list_of_weights: List[float])->None:
        """Take a list of weights and set the
        neural network weights according to these weights"""
        # Check the size
        if len(list_of_weights) != self.num_weights:
            raise Exception("Weights are being set incorrectly in setWeights().\n"\
                            "The number of weights in the list is not the same as\n"\
                            "the number of weights in the network\n"\
                            +str(len(list_of_weights))+"!="+str(self.num_weights))
        list_ind = 0
        for layer in self.weights:
            for row in layer:
                for element_ind in range(row.size):
                    row[element_ind] = list_of_weights[list_ind]
                    list_ind+=1

    def getWeights(self) -> List[float]:
        """Get the weights as a list"""
        weight_list = []
        for layer in self.weights:
            for row in layer:
                for element in row:
                    weight_list.append(element)
        return weight_list

    def activation(self, arr: np.ndarray) -> np.ndarray:
        return np.tanh(arr)

    def calculateNumWeights(self) -> int:
        return sum([w.size for w in self.weights])
