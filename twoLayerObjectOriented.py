
import numpy as np
'''import sys
print(sys.version)
print(sys.prefix)'''

np.random.seed(0)

#inputs are "X"
X = [[   1,   2,    3,  2.5],
	 [ 2.0, 5.0, -1.0,  2.0],
	 [-1.5, 2.7,  3.3, -0.8]]
	

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		#(rows, columns)
		#*0.10 to keep values between -1 and 1
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		#biases creates a 1 row, n column array.
		#the first parameter of zeros is the matrix shape, so we need to pass it as param 
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

# row length [n_inputs], neurons [whatever you want],
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)