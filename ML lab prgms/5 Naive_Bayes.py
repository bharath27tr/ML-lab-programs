import csv
import random
import math
import operator
import numpy as np
from collections import Counter, defaultdict

def loadCsv(filename):
	lines = csv.reader(open(filename))
	dataset = list(lines)
    for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset
 
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
        copy = list(dataset)
        i=0
        while len(trainSet) < trainSize:
		#index = random.randrange(len(copy))
		
		trainSet.append(copy.pop(i))
        return [trainSet, copy]

def occurrences(list1):
    no_of_examples = len(list1)
  #  print no_of_examples
   # print Counter(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob

def naive_bayes(training,outcome, test_data):
    
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
   
    likelihood_total={}
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
    for cls in classes:   
    	likelihood_total[cls] = defaultdict(list)
    
    class_probabilities = occurrences(outcome)
   
    for cls in classes:   
        for j in range(0,cols):
	     likelihood_total[cls][j] += list(training[:,j])
   
    for cls in classes:
        for j in range(0,cols):
             likelihood_total[cls][j] = occurrences(likelihood_total[cls][j])   
   
        
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
             likelihoods[cls][j] += list(subset[:,j])
       

    for cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = occurrences(likelihoods[cls][j])

       
    h=[]
    prob=[]
    results={}
    rows, cols  = np.shape(test_data)
    for ins in range(0,rows):
        results[ins] = defaultdict(dict)
    
    for j in range(0,rows):
     for cls in classes:
         class_probability = class_probabilities[cls]
         data_probability=1.0
         
         for i in range(0,len(test_data)):
              	relative_values = likelihoods[cls][i]
             	rv_total = likelihood_total[cls][i]
	     	if test_data[j][i] in relative_values.keys():
             	    class_probability *= relative_values[test_data[j][i]]
		    data_probability *= rv_total[test_data[j][i]]
              	else:
                    class_probability *= 0
                results[j][cls]=class_probability/data_probability
    prob.append([np.argmax(results[v].values()) for v in results.keys()])
   
    for j in range(0,cols):
	if prob[0][j]==0:
		prob[0][j]=10.0
        else:
                prob[0][j]=5.0
    return prob[0]


#Get Accuracy

def getAccuracy(actual, predictions):

	correct = 0

	for x in range(len(actual)):

		if actual[x] == predictions[x]:

			correct += 1

	return (correct/float(len(actual)))*100.0

   
def main():
	filename = 'ConceptLearning.csv'
	splitRatio = 0.75
	dataset = loadCsv(filename)
	trainingS, testSet = splitDataset(dataset, splitRatio)
        training=[]
        outcome=[]
        i=0
        for xcx in trainingS:
		i = i + 1
	        training.append(xcx[:-1])
                outcome.append(xcx[-1])
	       # print trainingS
        training=np.asarray(training)
        #training=tr
	i=0
	test=[]
	testoutcome=[]
        for xcx in testSet:
		i = i + 1
	        test.append(xcx[:-1])
                testoutcome.append(xcx[-1])
	testdata=np.asarray(test)
        print('Split {0} rows into'.format(len(dataset)))
	print('Number of Training data: ' + (repr(len(training))))
	print('Number of Test Data: ' + (repr(len(testSet))))
	print("\nThe values assumed for the concept learning attributes are\n")
	print("OUTLOOK=> Sunny=1 Overcast=2 Rain=3\nTEMPERATURE=> Hot=1 Mild=2 Cool=3\nHUMIDITY=> High=1 Normal=2\nWIND=> Weak=1 Strong=2")
	print("TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5")
	print("\nThe Training set are:")
	for x in trainingS:
		print(x)
	print("\nThe Test data set are:")
	for x in testSet:
		print(x)
	print("\n")
        probabilities=naive_bayes(training,outcome,testdata)
        actual = []
	for i in range(len(testSet)):
		vector = testSet[i]
		actual.append(vector[-1])
	print('Actual values: {0}%'.format(actual))
        print('Prediction values: {0}%'.format(probabilities))
	accuracy = getAccuracy(actual, probabilities)

	print('Accuracy: {0}%'.format(accuracy))


main()

