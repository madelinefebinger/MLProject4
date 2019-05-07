import random

class Perceptron: 
	def __init__(self,training_examples,weights,target_concept):
		self.training_examples = training_examples
		self.weights = weights
		self.target_concept = target_concept

		self.error_history = []
		self.results = []

		self.perceptron_learning_alg(training_examples,weights)

		#num_weights = len(training_examples[0]) - 1
		#self.weights = self.get_random_starting_weights(num_weights)

	# Perceptron learning algorithm
	def perceptron_learning_alg(self, training_examples,weights):
		learning_rate = 0.01
		errors = self.get_num_errors(training_examples,weights) # Get initial error
		self.error_history.append(errors)
		self.results.append(list(weights))

		while (errors > 0): # TODO: check if error stops decreasing
			errors = 0
			# Iteratively apply the perceptron to each training ex. 
			for example in training_examples:
				# Get target output and actual output
				t = self.get_target_output(example)
				o = self.output(weights,example)

				if (t != o):
					errors += 1

				# Update weights
				for i in range(len(weights)): 
					x_i = float(example[i])
					w_i = weights[i]
					delta_w_i = learning_rate * (t - o) * x_i
					weights[i] += delta_w_i
			print(errors)
			self.error_history.append(errors)
			self.results.append(list(weights)) # Save results for this epoch 
		print ("Final Weights:", weights)


	# Returns output of the perceptron unit
	def output(self,weights,example):
		sum = 0

		for i in range(len(weights)):
			x = float(example[i])
			w = weights[i]
			sum += w*x

		if (sum > 0):
			return 1
		else:
			return -1


	# Create a random starting weight vector for the perceptron
	def get_random_starting_weights(self,num_weights): 
		weights = []

		for i in range(num_weights):
			weight = random.uniform(-1,1)
			weights.append(weight)

		return weights

	# Returns the number of errors with the given weights
	def get_num_errors(self,training_examples,weights):
		num_errors = 0

		for example in training_examples:
			output = self.output(weights, example)
			target_output = self.get_target_output(example)

			if (output != target_output):
				num_errors = num_errors + 1

		print(num_errors)
		return num_errors

	# Returns the target output for the given example
	def get_target_output(self, example):
		if (example[-1] == self.target_concept):
			return 1
		else:
			return -1

		