"""
CMPE 452 Assignment 1
Curtis Shewchuk
14cms13
A simple 7 input perceptron network that is used to classify three different kinds of wheat.
We use a 3 by 7 matrix of weights that connect the 7 input layers to the three output perceptron neurons. The 7 input
data points are multiplied by the weight matrix (denoted input_kernels) to produce a 1x3 vector, with each index representing the 
weighted sum calculated by the perceptron. This vector is passed through the activation function which then outputs zeros
"""

import numpy as np
import perceptron as pc
## create inputs array and load in test/traning data. Inputs contains all the weights between the input node and each output node
filename = 'networkoutput.txt'
ifile = open(filename, 'w')
ifile.write('Starting Kernels\n')
input_kernels = pc.generate_random_start_weights(7,3)
# Write the starting random weights to the text file
ifile.write("\n".join(str(elem) for elem in input_kernels))
previous_run_kernels = pc.generate_random_start_weights(7,3)
data = np.genfromtxt('trainSeeds.csv', delimiter= ',')
testData = np.genfromtxt('testSeeds.csv', delimiter=',')

### Training the Network ###
## Create the arrays for error checking and the max iterations that the training shoud take.
iteration  = 0
max_iteration = 6500
expected_output = 0
error_square_array = np.zeros(3)
old_error_array = [200,200,200] # the original errors must be large, so that it at least does more than one loop of training
error_threshold = 5
error_flag_array = [False, False, False] # After each iteration, this array is checked. If they are ALL set to true, the loop breaks, and training halts
## training of the network
while iteration < max_iteration:
    # Train the Weights for an iteration of the training points
    for index in range(len(data)):
        strip =  data[index] #strip off current data points
        strip = list(strip) #the single value that indicates which output neuron should fire, determining which type of grain it is, spits a 1,2,3
        expected_output = strip.pop() # remove the expected firing neuron from the data set
        strip = np.array(strip) # return to a numpy array
        weighted_sum_output = np.matmul(strip,input_kernels)
        outputs = pc.activation_function(weighted_sum_output)
        input_kernels= pc.check_for_correct_output_train(outputs, weighted_sum_output, input_kernels, expected_output, strip)

    #Error check after each iteration of weight training
    for index in range(len(data)):
        strip = data[index]  # strip off current data points
        strip = list(strip)  # the single value that indicates which output neuron should fire, determining which type of grain it is, spits a 1,2,3
        expected_output = strip.pop()
        # Create the array to check the output. Since the popped value will be a 1,2, or 3, the index which we need to check
        # in the expected output array is the popped value - 1 (so if index 0 has a value of 1, then the first classifying perceptron fired)
        expected_output_array = np.zeros(3)
        expected_output_array[int(expected_output) - 1] = 1 
        strip = np.array(strip)
        weighted_sum_output = np.matmul(strip, input_kernels)
        outputs = pc.activation_function(weighted_sum_output)
        
        ## Compound the error after each iteration
        for i in range(len(error_square_array)):
            error_square_array[i] = error_square_array[i] + (expected_output_array[i] - outputs[i])**2

    #Now loop over error_square_array and check if any error is above the threshold
    for i in range(len(error_square_array)):
        if old_error_array[i] + error_threshold < error_square_array[i]:
            error_flag_array[i] = True
        elif error_square_array[i] == 0:
            error_flag_array[i] = True

    #Break the while loop if all weights are below the threshold
    if error_flag_array[0] == error_flag_array[1] == error_flag_array[2] == True:
        print iteration
        input_kernels = previous_run_kernels
        break
    # if not over the threshold, reset the error arrays
    else:
        old_error_array = error_square_array
        error_square_array = np.zeros(3)
        previous_run_kernels = input_kernels
        error_flag_array = [False, False, False]
    iteration += 1

ifile.write('\nEnding kernel weights after training\n')
ifile.write("\n".join(str(elem) for elem in input_kernels))
errors = 0
outputs_matrix = [[0 for _ in range(4)] for _ in range(3)]

### TESTING THE TRAINED DATA SET ###
for index in range(len(testData)):
    strip = testData[index]  # strip off current data points
    strip = list(strip)  # the single value that indicates which output neuron should fire, determining which type of grain it is, spits a 1,2,3
    expected_output = strip.pop()
    strip = np.array(strip)
    weighted_sum_output = np.matmul(strip, input_kernels)
    outputs =  pc.activation_function(weighted_sum_output)
    index_string = str(index + 1)
    ifile.write('\nTest Data Point ' + index_string)
    expected_output = pc.check_for_correct_output_test(outputs,expected_output, ifile)
    outputs_matrix = pc.calculate_positive_negative(outputs, expected_output, outputs_matrix)

## PRECISION AND RECALL CALCULATIONS
outputs_matrix = list(outputs_matrix)
precision_one = float(outputs_matrix[0][0]) / float((outputs_matrix[0][0] + outputs_matrix[0][1]))*100
precision_two = float(outputs_matrix[1][0]) / float((outputs_matrix[1][0] + outputs_matrix[1][1]))*100
precision_three = float(outputs_matrix[2][0]) / float((outputs_matrix[2][0] + outputs_matrix[2][1]))*100
recall_one = float(outputs_matrix[0][0]) / float((outputs_matrix[0][0] + outputs_matrix[0][2]))*100
recall_two = float(outputs_matrix[1][0]) / float((outputs_matrix[1][0] + outputs_matrix[1][2]))*100
recall_three = float(outputs_matrix[2][0]) / float((outputs_matrix[2][0] + outputs_matrix[2][2]))*100

ifile.write('\nPrecisions and Recalls\n First Wheat Class\n')
ifile.write('Precision: ' + str(precision_one) + ' Recall: ' + str(recall_one))
ifile.write('\nSecond Wheat Class\n')
ifile.write('Precision: ' + str(precision_two) + ' Recall: ' + str(recall_two))
ifile.write('\nThird Wheat Class\n')
ifile.write('Precision: ' + str(precision_three) + ' Recall: ' + str(recall_three))

#confusion matrix
ifile.write("\nCONFUSION MATRICIES\n")
for index in range(len(outputs_matrix)):
    ifile.write("\nWheat Class " + str(index+1))
    ifile.write("\n TP: " + str(outputs_matrix[index][0]) + " FP: " + str(outputs_matrix[index][1]))
    ifile.write("\n FN: " + str(outputs_matrix[index][2]) + " TN: " + str(outputs_matrix[index][3]))

ifile.close()

