import csv
import perceptron
import matplotlib.pyplot as pyplot

training_examples = []

with open('iris.data') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',')
	for row in csv_reader:
		row.insert(0,1) # add constant x_0 = 1 
		training_examples.append(row)

# Task 2 
initial_weights = [0,0,0,0,0]
print("LP1: Iris-setosa")
p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
print(p1.results)
print(p1.error_history)


"""
initial_weights = [0,0,0,0,0]
#The error is never reaching 0 on this one so I commented it out for now.
print("LP2: Iris-versicolor")
p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
print(p2.results)
"""

initial_weights = [0,0,0,0,0]
print("LP3: Iris-virginia")
p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginia")
print(p3.results)
print(p3.error_history)