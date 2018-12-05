import numpy as np
import math
from dataloader import read_data

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        
    def __str__(self):
        return self.attribute

def datasubset(data, col, delete):
    dict = {}
    values=[]
    values = np.unique(data[:, col])
    count = np.zeros((values.shape[0], 1), dtype=np.int32)    
    #count the total number of training examples matching with values
    for x in range(values.shape[0]):
        for y in range(data.shape[0]):
            if data[y, col] == values[x]:
                count[x] += 1
    #create a subset            
    for x in range(values.shape[0]):
        dict[values[x]] = np.empty((count[x], data.shape[1]), dtype="|S32")   
	pos = 0
        for y in range(data.shape[0]):
            if data[y, col] == values[x]:
                dict[values[x]][pos] = data[y]
		pos += 1       
	
        if delete:
            dict[values[x]] = np.delete(dict[values[x]], col, 1)
    
    return values, dict    
   
def entropy(S):
    values = np.unique(S)
    if values.size == 1:
        return 0
    
    counts = np.zeros((values.shape[0], 1))
    sums = 0
    
    for x in range(values.shape[0]):
	counts[x] = sum(S == values[x]) / (S.size * 1.0)
	
    for count in counts:
        sums += -1 * count * math.log(count, 2)
    return sums
    
def gain_ratio(data, col):
    values, dict = datasubset(data, col, delete=False)          
    total_size = data.shape[0]
    entropies = np.zeros((values.shape[0], 1))
           
    for x in range(values.shape[0]):
        ratio = dict[values[x]].shape[0]/(total_size * 1.0)
	entropies[x] = ratio * entropy(dict[values[x]][:, -1])
 
    total_entropy = entropy(data[:, -1])
       
    for x in range(entropies.shape[0]):
        total_entropy -= entropies[x]
    return total_entropy 

def create_node(data, metadata):

    if (np.unique(data[:, -1])).shape[0] == 1:
        node = Node("")
        node.answer = np.unique(data[:, -1])[0]
	return node
    
    gains = np.zeros((data.shape[1] - 1, 1))
   
    for col in range(data.shape[1] - 1):
        gains[col] = gain_ratio(data, col)
  
    split = np.argmax(gains)

    node = Node(metadata[split])    
   
    metadata = np.delete(metadata, split, 0)    
    values,dict = datasubset(data, split, delete=True)
   
    for x in range(values.shape[0]):
	child = create_node(dict[values[x]], metadata)
        node.children.append((values[x], child))
    
    return node        
    
def empty(size):
    s = ""
    for x in range(size):
        s += "   "
    return s

def print_tree(node, level):
    if node.answer != "":
        print empty(level), node.answer
        return
        
    print empty(level), node.attribute
    
    for value, n in node.children:
        print empty(level + 1), value
        print_tree(n, level + 2)
        
metadata, traindata = read_data("input1.csv")

data = np.array(traindata)

node = create_node(data, metadata)
    
print_tree(node, 0)
