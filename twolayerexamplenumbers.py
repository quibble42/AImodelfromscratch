
import numpy as np
import sys
print(sys.version)
print(sys.prefix)
inputs = [[1,2,3,2.5],
		  [2.0, 5.0, -1.0, 2.0],
		  [-1.5,2.7, 3.3, -0.8]]
	

weights = [[0.2,0.8,-0.5, 1.0],
 			[0.5, -0.91, 0.26, -0.5],
			[-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]


weights2 = [[0.1,-0.14, 0.5,],
 			[-0.5, 0.12, -0.33],
			[-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

#The below commented code is the array math being done in base python, without numpy.
''' output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
		 inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
		 inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

layer_outputs = []
for nueron_weights, neuron_bias in zip(weights, biases):
	neuron_output = 0
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output += n_input*weightsneuron_output 
		neuron_output += neuron_bias
		layer_outputs.append(neuron_output)

print(layer_outputs)
'''

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)