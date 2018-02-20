import mnist_loader
import network2

training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()

cost = network2.QuadraticCost();

net = network2.Network([784, 50, 10], cost)

net.SGD(training_data, 75, 8, 3.0, 0.0, evaluation_data=test_data, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)