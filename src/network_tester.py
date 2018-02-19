"""
network_tester.py
~~~~~~~~~~~~~~~~

A module to implement the network.py class.
"""

import mnist_loader
import network
from datetime import datetime, date
import logger

# NETWORK 1 IMPLEMENTATION

log_file_name = "DataCollected/Number of Layers H100B8.txt"
log = logger.Logger(log_file_name, True)

for x in range(0, 5):
    if x == 0:
        networkLayout = [784, 50, 50, 10]
        number_of_epochs = 75
        mini_batch_size = 8
        learning_rate = 3.0
        number_of_runs = 5

    elif x == 1:
        networkLayout = [784, 60, 40, 10]
        number_of_epochs = 75
        mini_batch_size = 8
        learning_rate = 3.0
        number_of_runs = 5

    elif x == 2:
        networkLayout = [784, 75, 25, 10]
        number_of_epochs = 75
        mini_batch_size = 8
        learning_rate = 3.0
        number_of_runs = 5

    elif x == 3:
        networkLayout = [784, 50, 30, 20, 10]
        number_of_epochs = 75
        mini_batch_size = 8
        learning_rate = 3.0
        number_of_runs = 5

    elif x == 4:
        networkLayout = [784, 25, 25, 25, 25, 10]
        number_of_epochs = 75
        mini_batch_size = 8
        learning_rate = 3.0
        number_of_runs = 5

    log.log_line("\n\nSet: " + str(x))
    print "Set: " + str(x)

    log.log_line('\n\t\tNetwork Details\nNetwork layout: '+str(networkLayout)+ '\nNumber of Epochs: '+
                 str(number_of_epochs)+'\nBatch Size: '+str(mini_batch_size)+
                 '\nLearning Rate: '+str(learning_rate)+'\nNumber of Runs: '+str(number_of_runs))

    for x in range(0, number_of_runs):
        log.log_line("\n\nIteration: "+str(x))
        print "Iteration: "+str(x)

        # Start timestamp
        startTime = datetime.today()
        log.log_line('\nStart Time:'+str(startTime))

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
        net = network.Network(networkLayout, log)

        # Stochastic Gradient Descent

        net.SGD(training_data, number_of_epochs, mini_batch_size, learning_rate, test_data=test_data,
                validation_data=validation_data)

        #  End Timestamp
        endTime = datetime.today()

        log.log_line('\nEnd Time:'+str(endTime))

log.close()
