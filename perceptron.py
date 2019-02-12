"""
CMPE 452 Assignment 1
Curtis Shewchuk
14cms13

Python file that contains some of the function that are called in the a1_452.py, the bulk of
my solution to assignment 1.
"""
import random
import numpy as np
training_factor = 0.85 #'c'
global previous_error
global errors
## we check if each neuron fired correctly based on the expected output. Then correct the weights if not
def check_for_correct_output_train(calulated_output, weighted_sum, weight_array, expected_output, inputs):
    """
    Checks if the calculated output is the same as the expected output, and if not, updates the weights for that
    specific neuron. 
    """
    output = np.zeros(3)
    output[int(expected_output)-1] = 1
    # loop over the output array, check which if the output was correct, and if not, update the weights according
    for index in range(len(calulated_output)):
        if (calulated_output[index] != output[index]):
            factor = 1
            if(weighted_sum[index] > output[index]):
                factor = -1
            weight_array = update_weights(weight_array, index, factor, inputs)
    return weight_array

# update the weights at the given index.
def update_weights(weight_array, index, factor, inputs):
    for val in range(len(weight_array)):
        weight_array[val][index] += training_factor*factor*inputs[val]
    return weight_array

def activation_function(weighted_sums):
    i = 0
    for sum in weighted_sums:
        if (sum >= 0):
            weighted_sums[i] = 1
        else:
            weighted_sums[i] = 0
        i += 1

    return weighted_sums


def check_for_correct_output_test(outputs, expected_output, ifile):
    """
    Similar Function to check_correct_output_train
    Calculates whether the calculated outputs match the expected output, and writes to the text file
    """
    output = np.zeros(3)
    output[int(expected_output) - 1] = 1
    total_incorrect = 0
    for i in range(len(outputs)):
        if(output[i] != outputs[i]):
            total_incorrect += 1

    if(total_incorrect > 0):
        print "Did not correctly fire"
        ifile.write('\nDid not fire correctly')
    else:
        print "Correct output"
        ifile.write('\nCorrect Output')
    print "Calculated Output"
    print outputs
    print "Expected Output"
    print output
    ifile.write('\nCalculated Output\n')
    ifile.write(" ".join(str(elem) for elem in outputs))
    ifile.write('\nExpected Output\n')
    ifile.write(" ".join(str(elem) for elem in output))
    return output

def generate_random_start_weights(rows, columns):
    random.seed()
    weights = np.zeros((rows,columns))
    for row in range(rows):
        for column in range(columns):
            factor = random.randint(0,10) % 2
            if factor == 0:
                factor = -1
            else:
                factor = 1
            weights[row][column] = factor*(random.randint(1,10) % 100)
    return weights


def calculate_positive_negative(calculated_output, expected_output, outputs_matrix):
    """
    Takes in the calculated outputs, and expected outputs, and increments the confusion matrix for TP, FP, FN, and TN
    Outputs matrix
    Index 0: True  Positive
    Index 1: False Positive
    Index 2: False Negative
    Index 3: True Negative
    :return
    """
    for i in range(len(calculated_output)):
        if calculated_output[i] != expected_output[i]:
            if(calculated_output[i] == 1):
                outputs_matrix[i][1] +=1
            else:
                outputs_matrix[i][2]+=1
        else:
            if calculated_output[i] == 0:
                outputs_matrix[i][3] += 1
            else:
                outputs_matrix[i][0] += 1
    return outputs_matrix



