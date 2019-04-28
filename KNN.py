from sklearn.datasets import load_iris
import math
from collections import Counter


def load_data ():
    iris_data = load_iris()
    X = iris_data["data"]
    y = iris_data["target"]
    return X,y

def create_train_test(percent, X, y):
    permutation = np.random.permutation(len(y))
    X = X[permutation]
    y = y[permutation]
    split_index = int(len(y) * percent)
    train = (X[:split_index, :], y[:split_index])
    test = (X[split_index:, :], y[split_index:])
    return train, test

def distance_calc(instance1,instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def get_neighbors(train, test_instance, k):
    distances = []
    for x in range(len(train[0])):
        dist_x=distance_calc(train[0][x], test_instance, len(test_instance)-1)
        distances.append((train[0][x],train[1][x], dist_x))
    distances=sorted(distances, key=lambda x: x[2])

    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][1])
    print ("k neighbors for test instance {} are: {}".format(test_instance,neighbors))
    return neighbors

def get_mode_value(neighbors):
    data=Counter(neighbors)
    mode=data.most_common(1)
    return (mode[0][0])

def get_accuracy(test, predictions):
	correct = 0
	for x in range(len(test[0])):
		if test[1][x] == predictions[x][-1]:
			correct += 1
	return (correct/float(len(test[0]))) * 100.0


def main():
    k=3
    X,y=load_data()
    train, test = create_train_test(0.7, X, y)

    predictions =[]
    for x in range(len(test[0])):
        neighbors = get_neighbors(train,test[0][x],k)
        y_value = get_mode_value(neighbors)
        predictions.append((test[0][x],y_value))

    print(predictions)
    accuracy = get_accuracy(test, predictions)
    print ("For K={}, Accuracy is {}%".format(k,accuracy))

main()

