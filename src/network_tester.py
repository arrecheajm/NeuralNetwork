"""
network_tester.py
~~~~~~~~~~~~~~~~

A module to implement the network.py class.
"""

import mnist_loader
import network
import datetime
import logger

# NETWORK 1 IMPLEMENTATION

log_file_name = "Batch Size Test"

for x in range(0, 5):
    if x == 0:
        mini_batch_size = 8
    elif x == 1:
        mini_batch_size = 16
    elif x == 2:
        mini_batch_size = 32
    elif x == 3:
        mini_batch_size = 64
    elif x == 4:
        mini_batch_size = 128

    # Network layers
    input_layer = 784
    hidden_layer = 100
    output_layer = 10

    # Training Data
    number_of_epochs = 75
    # mini_batch_size = 10
    learning_rate = 3.0
    number_of_runs = 5

    log = logger.Logger(log_file_name, True)

    # Start timestamp
    log.log_line('Network Details:\n\nNumber of Layers: '+str(3)+'\nNumber of Neurons in Hidden Layer: '+str(hidden_layer)+\
          '\nNumber of Epochs: '+str(number_of_epochs)+'\nBatch Size: '+str(mini_batch_size)+\
          '\nLearning Rate:'+str(learning_rate)+'\nNumber of Runs: '+str(number_of_runs))

    for x in range(0, number_of_runs):
        log.log_line("\n\nRun: "+str(x))
        log.log_line('\nStart Time:'+str(datetime.datetime.now().time()))

        # Loading MNIST data
            # training_data: Training input and desired outputs
            # validation_data: Used to set hyper-parameters of the network
            # test_data: Used to evaluate the output of every epoch #
        training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()

        # Network Setup
            # input_layer: Number of neurons in first layer
            # hidden_layer: Number of neurons in the hidden layer
            # output_layer: Number of neurons in the output layer
        net = network.Network([input_layer, hidden_layer, output_layer], log)

        # Stochastic Gradient Descent

        net.SGD(training_data, number_of_epochs, mini_batch_size, learning_rate, test_data=test_data)

        #  End Timestamp
        log.log_line('\nEnd Time: '+str(datetime.datetime.now().time()))

log.close()

#  NETWORK 2 IMPLEMENTATION
