import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np

class Network: 
    
    # instance variables
    nodes = []	# an array of numbers of nodes
    n_layers = 0	# determined by lentgh of nodes array
#     n_hidden = 0 # n_layers - 2
    weights = []	# array of each weight matrix, len(weights) = num_layers -1
    biases = []	# array of each bias vector, len(biases) = num_layers -1
    losses = []	# initialized as empty array, updated through training
    train_accuracies = []	# initialized as empty array, updated through training
    test_accuracies = []	# initialized as empty array, updated through training
    test_num = []	# initialized as empty array, updated through training
    output = []	# initialized as empty array, updated through training
    loss = [] 	# current loss value, train_labels - output
    wasTested = False	# boolean which sais whether or not the network was tested while training, used for plotting
    layers = [] # layer vals
    
    #     n_epochs	# number of epochs, inputted as train param
    lr = 0.01 # learning rate, train parameter
    #     train_data = []	# inputted as train param
    #     train_labels = []	# inputted as train param
    #     test_data = []	# inputted as train param
    #     test_labels = []	## inputted as train param

    def get_output(self):
        return self.layers[-1]
    
    def get_accuracy(self, _data, _labels):
        m = _labels.shape[0]
        pred = (self.pred(_data))[-1] # last index of layer vals
        print("pred shape", pred.shape)
        print("data shape:", _data.shape)
        print("label shape:", _labels.shape)
        pred = pred.reshape(_labels.shape)
        error = torch.sum(torch.abs(pred.float()-_labels.float()))
        return ((m-error)/m) * 100
    
    # methods
    def __init__(self, _nodes):
        self.nodes = _nodes # set array of nodes
        self.num_layers = len(_nodes)
#         n_hidden = len(_nodes)-2
            
        for w in self.weights:
            print("weight shape:", w.shape)
    
        # initialize weights 
        self.weights = []
        for i in range(len(self.nodes)-1):
            w = torch.randn(self.nodes[i], self.nodes[i+1])
            self.weights.append(w)
        # initialize biases
        self.biases = []
        for i in range(len(self.nodes)-1):
            b = torch.randn(1, self.nodes[i+1])
            self.biases.append(b)   
            
    ## sigmoid activation function using pytorch
    def sigmoid_activation(self, z):
        return 1 / (1 + torch.exp(-z))    
    
    def pred(self, _data):
        layer_vals = []
        a = _data
        a = a.float()
#         print("a is \n", a)
        for i in range(len(self.nodes)-2): # feed thru hidden layers
            print(self.weights[i])
#             wt = self.weights[i].clone()
            print("a shape:", a.shape)
            print("weights", i, "shape:", self.weights[i].shape)
            z = torch.mm(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid_activation(z)
            layer_vals.append(a)
        z_out = torch.mm(a, self.weights[-1]) + self.biases[-1]
        output = self.sigmoid_activation(z_out)
        layer_vals.append(output)
    #     print(output)
        return layer_vals 
        
    def forward_prop(self, _data):
        self.layers = self.pred(_data)
    
    # loss computation
    # loss = y - output
    def calculate_loss(self, _labels):
        return _labels.float() - self.layers[-1].float()
    
    ## function to calculate the derivative of activation
    def sigmoid_delta(self, x):
        return x * (1 - x)

    def backprop_and_update(self, _data, _labels):
        loss = self.calculate_loss(_labels)
        deltas = [] # from first hidden to output
        ds = [] # from output to first hidden
        # compute derivative of error terms
        for a in self.layers:
            deltas.append(self.sigmoid_delta(a))
        loss_d = loss
        for d, w in zip(reversed(deltas), reversed(self.weights)):
            dd = loss * d
            loss_d = torch.mm(d, w.t())
            ds.append(dd)
        _as = copy.deepcopy(self.layers)
        _as.insert(0,_data) # insert input into the layer vals
        del _as[-1]
    #     print("\n Layer vals \n", _layer_vals)
        # Update parameters
        new_weights = []
        new_biases = []
        print("lengths", len(ds), len(self.weights), len(_as))
        for d, w, a, b in zip(ds, reversed(self.weights), _as, reversed(self.biases)):
            wt = torch.mm(a.t().float(),d.float()) * self.lr
            new_weights.insert(0, wt)

            bi = b + d.sum()*self.lr
            new_biases.insert(0,bi)
        self.weights = new_weights
        self.biases = new_biases
        

    def train(self, _train_data, _train_labels, _n_epochs, _lr=0.01, _test=False, _test_data=[], _test_labels=[]):
        self.lr = _lr
        for i in range(_n_epochs):
            self.forward_prop(_train_data)
            # test here 
            if (i%50 == 0):
                print("train accuracy is:", self.get_accuracy(_train_data, _train_labels))
                if(_test==True):
                    print("train accuracy is:", self.get_accuracy(_test_data, _test_labels))
            self.backprop_and_update(_train_data, _train_labels)

        


        
    
    