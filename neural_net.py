import numpy as np

class NeuralNetwork():

    #Sets our random seed and creates random synaptic values based off of it1
    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    #Returns the "normalized" version of x 
    #Between 0 and 1 using a logarithmic function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #Takes in a normalized value and recreates the original
    #I.E. reverts the conversion done in the sigmoid function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    #Loops as many times as desired with training_iterations
        #Uses think function to calculate error
        #Multiplies the inputs by the ( error by the output ) to decide the adjustments
        #Adjusts the synaptic weights by the adjustments
    def train(self, training_inputs, training_outputs, training_iterations):
        for interation in range(training_iterations):
            
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    #Gets all the normalized values and multiplies them by their weights
    #Returning an array output containing all new weighted values after "thinking"
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output

if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print('Random synaptic weights')
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,1],
                            [1,0,1],
                            [1,0,1],
                            [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    neural_network.train(training_inputs, training_outputs, 10000)

    print('Synaptic weights after training')
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print('new situation: input data = ', A, B, C)
    print('Output data:')

    print(neural_network.think(np.array([A, B, C])))