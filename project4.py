import csv
import perceptron
import matplotlib.pyplot as plt
import random

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

# Create a random starting weight vector for the perceptron
def get_random_starting_weights(): 
	weights = []

	for i in range(4):
		weight = random.uniform(0,1)
		weights.append(weight)

	return weights

def create_epoch_stats_file(perceptron,output_filename):
	f = open(output_filename, "w")
	results = perceptron.results
	error_history = perceptron.error_history

	for i in range(len(results)):
		epoch = str(i)
		f.write(epoch)
		f.write('	')
		error = str(error_history[i])
		f.write(error)
		f.write('	')
		weights = str(results[i])
		f.write(weights)
		f.write('\n')

	f.close()

# Task 2, Initial Weights all 0 
initial_weights = [0,0,0,0,0]
print("T2 LP1: Iris-setosa")
task2_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
#print(task2_p1.results)
#print(task2_p1.error_history)
graph(task2_p1.error_history,"T2 LP1: Iris-setosa","iris-setosa-t2.png")
create_epoch_stats_file(task2_p1,"epoch_stats_task2_lp1.txt")

initial_weights = [0,0,0,0,0]
print("T2 LP2: Iris-versicolor")
task2_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task2_p2.error_history,"T2 LP2: Iris-versicolor","iris-versicolor-t2.png")
create_epoch_stats_file(task2_p2,"epoch_stats_task2_lp2.txt")

initial_weights = [0,0,0,0,0]
print("T2 LP3: Iris-virginica")
task2_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task2_p3.error_history,"T2 LP3: Iris-virginica","iris-virginica-t2.png")
create_epoch_stats_file(task2_p3,"epoch_stats_task2_lp3.txt")

# Task 3.1, Initialize Weights all 1
initial_weights = [1,1,1,1,1]
print("T3.1 LP1: Iris-setosa")
task31_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
graph(task31_p1.error_history,"T3.1 LP1: Iris-setosa","iris-setosa-t31.png")
create_epoch_stats_file(task31_p1,"epoch_stats_task31_lp1.txt")

initial_weights = [1,1,1,1,1]
print("LP2: Iris-versicolor")
task31_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task31_p2.error_history,"T3.1 LP2: Iris-versicolor","iris-versicolor-t31.png")
create_epoch_stats_file(task2_p1,"epoch_stats_task31_lp2.txt")

initial_weights = [1,1,1,1,1]
print("T3.1 LP3: Iris-virginica")
task31_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task31_p3.error_history,"T3.1 LP3: Iris-virginica","iris-virginica-t31.png")
create_epoch_stats_file(task31_p3,"epoch_stats_task31_lp3.txt")

# Task 3.2 Initialize weights to 4 different numbers between 0 and 1
random_weights = get_random_starting_weights()

initial_weights = random_weights
print("T3.2 LP1: Iris-setosa")
task32_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
graph(task32_p1.error_history,"T3.2 LP1: Iris-setosa","iris-setosa-t32.png")
create_epoch_stats_file(task32_p1,"epoch_stats_task32_lp1.txt")

initial_weights = random_weights
print("T3.2 LP2: Iris-versicolor")
task32_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task32_p2.error_history,"T3.2 LP2: Iris-versicolor","iris-versicolor-t32.png")
create_epoch_stats_file(task32_p2,"epoch_stats_task32_lp2.txt")

initial_weights = random_weights
print("T3.2 LP3: Iris-virginica")
task32_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task32_p3.error_history,"T3.2 LP3: Iris-virginica","iris-virginica-t32.png")
create_epoch_stats_file(task32_p3,"epoch_stats_task32_lp3.txt")

# Task 3.3 Intialize weights to 4 different numbers between 
random_weights = get_random_starting_weights()

initial_weights = random_weights
print("T3.3 LP1: Iris-setosa")
task33_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
graph(task32_p1.error_history,"T3.3 LP1: Iris-setosa","iris-setosa-t33.png")
create_epoch_stats_file(task33_p1,"epoch_stats_task33_lp1.txt")

initial_weights = random_weights
print("T3.3LP2: Iris-versicolor")
task33_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task32_p2.error_history,"T3.3 LP2: Iris-versicolor","iris-versicolor-t33.png")
create_epoch_stats_file(task33_p2,"epoch_stats_task33_lp2.txt")

initial_weights = random_weights
print("T3.3 LP3: Iris-virginica")
task33_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task32_p3.error_history,"T3.3 LP3: Iris-virginica","iris-virginica-t33.png")
create_epoch_stats_file(task33_p3,"epoch_stats_task33_lp3.txt")

# Task 4, Shuffle training data
random.shuffle(training_examples)

# Task 4.1 Repeat Task 2 (initial weights all 0)
initial_weights = [0,0,0,0,0]
print("T4.1 LP1: Iris-setosa")
task41_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
graph(task41_p1.error_history,"T4.1 LP1: Iris-setosa","iris-setosa-t41.png")
create_epoch_stats_file(task41_p1,"epoch_stats_task41_lp1.txt")

initial_weights = [0,0,0,0,0]
print("T4.1 LP2: Iris-versicolor")
task41_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task41_p2.error_history,"T4.1 LP2: Iris-versicolor","iris-versicolor-t41.png")
create_epoch_stats_file(task41_p2,"epoch_stats_task41_lp2.txt")

initial_weights = [0,0,0,0,0]
print("T4.1 LP3: Iris-virginica")
task41_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task41_p3.error_history,"T4.1 LP3: Iris-virginica","iris-virginica-t41.png")
create_epoch_stats_file(task41_p3,"epoch_stats_task41_lp3.txt")

# Task 4.2 Repeat Task 2 with a different shuffling order
random.shuffle(training_examples)

initial_weights = [0,0,0,0,0]
print("T4.2 LP1: Iris-setosa")
task42_p1 = perceptron.Perceptron(training_examples,initial_weights,"Iris-setosa")
graph(task42_p1.error_history,"T4.2 LP1: Iris-setosa","iris-setosa-t42.png")
create_epoch_stats_file(task42_p1,"epoch_stats_task42_lp1.txt")

initial_weights = [0,0,0,0,0]
print("T4.2 LP2: Iris-versicolor")
task42_p2 = perceptron.Perceptron(training_examples,initial_weights,"Iris-versicolor")
graph(task42_p2.error_history,"T4.2 LP2: Iris-versicolor","iris-versicolor-t42.png")
create_epoch_stats_file(task42_p2,"epoch_stats_task42_lp2.txt")

initial_weights = [0,0,0,0,0]
print("T4.2 LP3: Iris-virginica")
task42_p3 = perceptron.Perceptron(training_examples,initial_weights,"Iris-virginica")
graph(task42_p3.error_history,"T4.1 LP3: Iris-virginica","iris-virginica-t42.png")
create_epoch_stats_file(task42_p3,"epoch_stats_task42_lp3.txt")