import numpy as np
import math
import random
import copy
import pickle
import csv

class Layer:
    def __init__(self):
        self.weightMatrix = []
        self.biases = []
        self.a = []
        self.delta = []

class Network:
    def __init__(self, layers):
        self.layers = layers

def createNetwork():
    layers = []
    input = Layer()
    layers.append(input)
    second = Layer()
    second.weightMatrix = 2*np.random.random((56,784))-1
    #print(second.weightMatrix)
    second.biases = 2*np.random.random((56,1))-1
    layers.append(second)
    third = Layer()
    third.weightMatrix = 2 * np.random.random((28, 56)) -1
    #print(third.weightMatrix)
    third.biases = 2*np.random.random((28,1))-1
    layers.append(third)
    fourth = Layer()
    fourth.weightMatrix = 2 * np.random.random((16, 28)) - 1
    #print(fourth.weightMatrix)
    fourth.biases = 2 * np.random.random((16, 1)) - 1
    layers.append(fourth)
    last = Layer()
    last.weightMatrix = 2* np.random.random((10, 16)) -1
    #print(last.weightMatrix)
    last.biases = 2*np.random.random((10,1))-1
    layers.append(last)
    return Network(layers)

def binNetwork(N):
    layers = []
    input = Layer()
    layers.append(input)
    second = Layer()
    second.weightMatrix = 2*np.random.random((4,N))-1
    #print(second.weightMatrix)
    second.biases = 2*np.random.random((4,1))-1
    layers.append(second)
    last = Layer()
    last.weightMatrix = 2 * np.random.random((1, 4)) -1
    #print(third.weightMatrix)
    last.biases = 2*np.random.random((1,1))-1
    layers.append(last)
    return Network(layers)

def xorNetwork():
    layers = []
    input = Layer()
    layers.append(input)
    second = Layer()
    second.weightMatrix = 2*np.random.random((2,2))-1
    #print(second.weightMatrix)
    second.biases = 2*np.random.random((2,1))-1
    layers.append(second)
    last = Layer()
    last.weightMatrix = 2 * np.random.random((1, 2)) -1
    #print(third.weightMatrix)
    last.biases = 2*np.random.random((1,1))-1
    layers.append(last)
    return Network(layers)


def binGenerate(N):
    a = []
    for i in range(2**N):
        b = bin(i)[2:]
        l = len(b)
        b = str(0) * (N - l) + b
        a.append(list(b))
    b = []
    for i in range(2**2**N):
        c = bin(i)[2:]
        l = len(c)
        c = str(0) * (2**N - l) + c
        b.append(list(c))
    trains = []
    for thing in b:
        train = []
        for i in range(2**N):
            train.append(np.array(list(thing[i])+a[i]))
        trains.append(np.array(train).astype(float))
    return trains
            
def circleData(N):
    train = []
    for i in range(N):
        x = 3*random.random()-3/2
        y = 3*random.random()-3/2
        if x**2 + y**2 <= 1:
            out = 1.
        else:
            out = 0.
        train.append(np.array([out,x,y]))
    return train

#train = circleData(60000)

#train = [np.array([0,0,0]),np.array([1,1,0]),np.array([1,0,1]),np.array([0,1,1])]
#train = np.array(train)
#network = binNetwork()
data = np.genfromtxt("mnisttrain.csv", delimiter = ",")
pickle.dump(data, open("mnisttrain.pk", "wb"))
data = np.genfromtxt("mnisttest.csv", delimiter = ",")
pickle.dump(data, open("mnisttest.pk", "wb"))
network = createNetwork()
pickle.dump(network, open("mnistnetwork.pk", "wb"))

#network = binNetwork(2)
#network = xorNetwork()
#train = [np.array([0,0,0]),np.array([1,1,0]),np.array([1,0,1]),np.array([0,1,1])]
#test = circleData(10000)
#for x in range(10):
#    train = circleData(60000)
#    print(x+1)
#    stochastic(train, network)
    #print(bintest(train, network))
#    print(bintest(test,network))

def transposeInput(array):
    matrix = []
    for n in array:
        matrix.append([n])
    return np.array(matrix)

def printDim(matrix):
    print(str(len(matrix)) + "x" + str(len(matrix[0])))

def sig(x, deriv = False):
    if(deriv):
        return x*(1-x)
    return 1/(1+np.exp(-np.array(x,np.longdouble)))


def forwardprop(network, input):
    network.layers[0].a = input
    for n in range(1,len(network.layers)):
        network.layers[n].a = sig(np.dot(network.layers[n].weightMatrix, network.layers[n-1].a)+network.layers[n].biases)
    return network.layers[len(network.layers)-1].a

def error(out, label):
    diff = []
    for y in range(len(out)):  #NUMBER SHOULD BE DIMENSION OF OUTPUT
        diff.append([out[y][0] - label])
        #if y == label:
        #    diff.append([out[y][0] - 1])
        #else:
        #    diff.append([out[y][0]])
    return np.array(diff)

def delta(network, i, point, label):
    if i == len(network.layers)-1:
        matrix1 = error(network.layers[i].a,label)
        matrix2 = sig(network.layers[i].a,True)
    else:
        matrix1 = np.dot(network.layers[i+1].weightMatrix.T, network.layers[i+1].delta)
        matrix2 = sig(network.layers[i].a, True)
    network.layers[i].delta = matrix1*matrix2
    return network.layers[i].delta

def backprop(network, point):
    label = point[0]
    array = np.delete(point, 0, axis=0)#/256
    input = transposeInput(array)
    actualOut = forwardprop(network, input)
    alpha = 0.01#valError(actualOut, label)*0.2
    weightGradient = []
    biasGradient = []
    for i in range(len(network.layers)-1):
        index = len(network.layers)-i-1
        if index != 1:
            d = np.dot(delta(network, index, input, label),network.layers[index-1].a.T)
        else:
            d = np.dot(delta(network, index, input, label), input.T)
        weightGradient.insert(0,d*alpha)
        biasGradient.insert(0, network.layers[index].delta*alpha)
    return (weightGradient, biasGradient)
def stochastic(train, init):
    data = pickle.load(open("mnisttrain.pk", "rb"))
    np.random.shuffle(data)
    network = pickle.load(open("mnistnetwork.pk", "rb"))
    count = 0
    for point in data:
        count += 1
        weightGradient, biasGradient = backprop(network, point)
        for i in range(1,len(network.layers)):
            network.layers[i].weightMatrix -= weightGradient[i-1]
            network.layers[i].biases -= biasGradient[i-1]
        if count % 600 == 0:
            print ("■"*(count//600)+" "*(100-count//600) + " ["+str(count//600)+"%"+"]", end="\r")
        pickle.dump(network, open("mnistnetwork.pk", "wb"))
    print()
    return network

def miniBatch(batchSize):
    data = pickle.load(open("mnisttrain.pk", "rb"))
    np.random.shuffle(data)
    network = pickle.load(open("mnistnetwork.pk", "rb"))
    count = 0
    totalWeightShift = []
    totalBiasShift = []
    for point in data:
        count += 1
        weightGradient, biasGradient = backprop(network, point)
        if totalWeightShift == [] and totalBiasShift == []:
            totalWeightShift = weightGradient
            totalBiasShift = biasGradient
        else:
            for i in range(len(network.layers)-1):
                totalWeightShift[i] += weightGradient[i]/batchSize
                totalBiasShift[i] += biasGradient[i]/batchSize
        if count % batchSize == 0:
            for i in range(1,len(network.layers)):
                network.layers[i].weightMatrix -= totalWeightShift[i-1]
                network.layers[i].biases -= totalBiasShift[i-1]

            totalWeightShift = []
            totalBiasShift = []
            print(int(count)//batchSize)
            pickle.dump(network, open("mnistnetwork.pk", "wb"))

def valError(matrix, label):
    sum = 0
    for n in range(len(matrix)): #NUMBER SHOULD BE DIMENSION OF OUTPUT
        sum += float((matrix[n][0]-label)**2/len(matrix))
        #if n == int(label):
        #    sum += float((matrix[n][0]-1.0)**2/len(matrix))#SAME WITH THESE ONES
        #else:
        #    sum += float((matrix[n][0]-0)**2/len(matrix))
    return sum

def test():
    data = pickle.load(open("mnisttrain.pk", "rb"))
    network = pickle.load(open("mnistnetwork.pk", "rb"))
    count = 0
    for i in range(60000):
        point = data[i]
        label = point[0]
        array = np.delete(copy.deepcopy(point), 0, axis=0)/256
        input = transposeInput(array)
        out = forwardprop(network, input)
        #print(out)
        max = 0
        maxIndex = -1
        for n in range(len(out)):
            if out[n][0] > max:
                max = out[n][0]
                maxIndex = n
        if int(label) != int(maxIndex):
            #print(i)
        #print("Expected:",int(label),"Actual:",int(maxIndex))
            #print("Error:", valError(out, label),"\\n")
            count += 1
    print(count)
    return count > 1500

def bintest(test, network):
    data = test
    count = 0
    for i in range(len(test)):
        point = data[i]
        label = point[0]
        array = np.delete(copy.deepcopy(point), 0, axis=0)
        input = transposeInput(array)
        out = forwardprop(network, input)
        #print(out)
        out = out[0][0]
        if abs(out-label)<0.5:
            count +=1
        #print(str(out) + " + " + str(label))
    #print(count)
    return count/len(test)

def display(array):
    for r in range(28):
        str = ""
        for c in range(28):
            if array[r*28+c][0] > 70/256:
                str += "■■"
            else:
                str += "  "
        print(str)
    print()
def finalTest():
    data = pickle.load(open("mnisttest.pk", "rb"))
    network = pickle.load(open("mnistnetwork.pk", "rb"))
    with open("mnistoutput.csv", "w") as csvfile:
        fieldnames = ["id", "number"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for i in range(1, len(data)):
            point = transposeInput(data[i]/256)
            point = np.delete(copy.deepcopy(point), 0, axis=0)
            out = forwardprop(network, point)
            max = 0
            maxIndex = -1
            for n in range(len(out)):
                if out[n][0] > max:
                    max = out[n][0]
                    maxIndex = n
            if max < 0.6:
                display(point)
                maxIndex = input(str(i) + ". What is this?")
            writer.writerow({"id": str(i), "number": maxIndex})

def update():
    network = pickle.load(open("mnistnetwork.pk", "rb"))
    pickle.dump(network, open("finalmnistnetwork.pk", "wb"))

def reset():
    network = pickle.load(open("finalmnistnetwork.pk", "rb"))
    pickle.dump(network, open("mnistnetwork.pk", "wb"))

def stats(network):
    #network = pickle.load(open("mnistnetwork.pk", "rb"))
    for layer in network.layers:
        if layer.weightMatrix == []:
            continue
        printDim(layer.weightMatrix)
        printDim(layer.biases)



        
#reset()
for x in range(5):
    stochastic('yee','haw')


#update()
#while test():
#    stochastic()
#miniBatch(32)
#test()
#stats()
#finalTest()
#output = np.genfromtxt("mnistoutput1.csv", delimiter = ",")
#pickle.dump(output, open("mnistoutput1.pk", "wb"))
