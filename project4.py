import csv
import perceptron
import matplotlib.pyplot as plt

training_examples = []

with open('iris.data') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',')
	for row in csv_reader:
		row.insert(0,1) # add constant x_0 = 1 
		training_examples.append(row)

def graph(error_history,plot_title,image_name):
	x = list(range(0,len(error_history)))
	y = error_history
	plt.title(plot_title)
	plt.xlabel('Epoch of Learning')
	plt.ylabel('Number of Errors')
	plt.scatter(x,y,s=50)
	x1,x2,y1,y2 = plt.axis()
	plt.axis((-1,x2,0,y2))
	plt.savefig(image_name)
	plt.clf()
	#plt.show()

# Task 2, Initial Weights all 0 
initial_weights = [0,0,0,0,0]
print("T2 LP1: Iris-setosa")
task2_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
#print(task2_p1.results)
#print(task2_p1.error_history)
graph(task2_p1.error_history,"T2 LP1: Iris-setosa","iris-setosa-t2.png")

initial_weights = [0,0,0,0,0]
print("LP2: Iris-versicolor")
task2_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task2_p2.error_history,"T2 LP2: Iris-versicolor","iris-versicolor-t2.png")

initial_weights = [0,0,0,0,0]
print("T2 LP3: Iris-virginia")
task2_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task2_p3.error_history,"T2 LP3: Iris-virginia","iris-virginica-t31.png")

# Task 3, Initial Weights all 1
initial_weights = [1,1,1,1,1]
print("T3.1 LP1: Iris-setosa")
task3_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
graph(task3_p1.error_history,"T3.1 LP1: Iris-setosa","iris-setosa-t31.png")

initial_weights = [1,1,1,1,1]
print("LP2: Iris-versicolor")
task3_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task3_p2.error_history,"T3.1 LP2: Iris-versicolor","iris-versicolor-t2.png")


initial_weights = [1,1,1,1,1]
print("T3.1 LP3: Iris-virginica")
task3_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task3_p3.error_history,"T3.1 LP3: Iris-virginica","iris-virginica-t31.png")