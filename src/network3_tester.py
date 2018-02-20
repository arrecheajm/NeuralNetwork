import mnist_loader
import network3

training_data, validation_data, test_data = \
            network3.load_data_shared()

input_layer = network3.SoftmaxLayer(784, 50)
hidden_layer = network3.SoftmaxLayer(50, 50)
output_layer = network3.SoftmaxLayer(50, 10)

net = network3.Network([input_layer, hidden_layer, output_layer], 8) # Takes a list of layers!

net.SGD(training_data, 75, 16, 3.0, validation_data, test_data, 0.0)