import mnist_loader
import network3

training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()

layer = network3.FullyConnectedLayer(784, 10)

net = network3.Network(layer, 8) # Takes a list of layers!

net.SGD(training_data, 75, 8, 3.0, validation_data=validation_data, test_data=test_data)