import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import pandas as pd

class Network: 
    
    # instance variables
    nodes = []	# an array of numbers of nodes
    n_layers = 0	# determined by lentgh of nodes array
#     n_hidden = 0 # n_layers - 2
    weights = []	# array of each weight matrix, len(weights) = num_layers -1
    w_initial = []
    weight_change = []
    biases = []	# array of each bias vector, len(biases) = num_layers -1
    b_initial = []
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
#         print("before",_data[:4],"\n", _labels[:4])
        m = _labels.shape[0] # number of labels
        pred = (self.pred(_data))[-1] # last index of layer vals
        pred = pred.numpy()
#         print("\n")
#         print("PREDICTED BEFORE", pred[:4])
#         print("PRED SIZE", len(pred))
#         print("SUM", np.sum(pred))
#         print("MAX", np.amax(pred))
#         print("pred shape", pred.shape)
#         print("data shape:", _data.shape)
#         print("label shape:", _labels.shape)
#         pred = pred.reshape(_labels.shape)
        pred[pred>.5] = 1
        pred[pred<=.5] = 0 
#         print("PREDICTED AFTER", pred[:4])
#         print("SUM AFTER", np.sum(pred))
#         print("MAX AFTER", np.amax(pred))
#         diff = _labels.numpy() - pred
#         print("LABELS", _labels.numpy())
#         print("SUM DIFF", np.sum(diff))
#         print("MAX DIFF", np.amax(diff))
#         abs_diff = np.abs(diff)
#         print("SUM ABS", np.sum(abs_diff))
#         print("MAX ABS", np.amax(abs_diff))
        error = np.sum(np.abs(_labels.numpy()-pred)) / len(_labels.numpy())
        if(error>1):
            print("ERROR GREATER THAN 1", len(_labels.numpy()), np.sum(pred), np.sum(np.abs(_labels.numpy()-pred)))
#             print("pred:", np.shape(pred), "labels:", np.shape(_labels))
#             print("ERROR IS GREATER THAN M. FUCK.")
#         print("after",_data[:4], "\n", _labels[:4])
        return ((1 - error)) *100
    
    # methods
    def __init__(self, _nodes, _init_weights=0, _init_biases=0):
#         np.random.seed(0) # doesn't work for some reason :(
        self.nodes = _nodes # set array of nodes
        self.num_layers = len(_nodes)
#         n_hidden = len(_nodes)-2
        if (_init_weights==0):   
            for w in self.weights:
                print("weight shape:", w.shape)

            weight_change = pd.DataFrame()
            # initialize weights 
            self.weights = []
            self.w_initial = []
            for i in range(len(self.nodes)-1):
                w = torch.randn(self.nodes[i], self.nodes[i+1])
                self.weights.append(w)
                self.w_initial.append(w)
    #             strg = 'W' + str(i)
    #             val = w
    #             weight_change[strg] = val
        else:
            self.weights = _init_weights
            self.w_initial = _init_weights

        if (_init_biases==0):
            # initialize biases
            self.biases = []
            self.b_initial = []
            for i in range(len(self.nodes)-1):
                b = torch.zeros(1, self.nodes[i+1])
                self.biases.append(b)   
                self.b_initial.append(b)
        else:
            self.biases = _init_biases
            self.b_initial = _init_biases
        
            
    ## sigmoid activation function using pytorch
    def sigmoid_activation(self, z):
        return 1 / (1 + torch.exp(-z))    
    
    
    def softmax_activation(self, z):
        exps = torch.exp(z-torch.max(z))
        return exps/torch.sum(exps)
    
    def softmax_delta(self, _labels):
        print("softmax delta haha")
    
    def pred(self, _data):
        layer_vals = []
        a = _data
        a = a.float()
#         print("a is \n", a)
        for i in range(len(self.nodes)-2): # feed thru hidden layers
#             print(self.weights[i])
#             wt = self.weights[i].clone()
#             print("a shape:", a.shape)
#             print("weights", i, "shape:", self.weights[i].shape)
#             print("a shape:", a.shape)
#             print("weights shape", self.weights[i].shape)
#             print("WEIGHT SHAPE FORWARD: \n", self.weights[i].shape)
#             print("ACTIVATION SHAPE FOR: \n", a.shape)
            z = torch.mm(a, self.weights[i]) + self.biases[i]
#             print("z shape:", z.shape)
            a = self.sigmoid_activation(z)
            layer_vals.append(a)
#         print("WEIGHT SHAPE OUT: \n", self.weights[-1].shape)
#         print("ACTIVATION SHAPE 2: \n", a.shape)
        z_out = torch.mm(a, self.weights[-1]) + self.biases[-1]
        activation_out = torch.nn.Softmax(dim=0)
        output = self.sigmoid_activation(z_out)
#         print(output)
#         print("OUTPUT SHAPE: \n", output.shape)
        layer_vals.append(output)
    #     print(output)
        return layer_vals 
        
    def forward_prop(self, _data):
        self.layers = self.pred(_data)
        
    def loss_derivative(self, _labels):
        return 2*(self.layers[-1].float() - _labels.float())
    
    # loss computation
    # loss = y - output
    def calculate_loss(self, _labels):
#         m = _labels.shape[0]
# # #         if(_labels.all() == 1):
# # #             return -1/m * torch.sum(torch.log(self.layers[-1].float()))
# # #         else: 
# #         return 1/m * torch.sum(torch.log(1 - self.layers[-1].float()))
        return 1/(len(_labels)) * torch.sum((_labels.float() - self.layers[-1].float())**2)
#         n_obs = len(_labels)
#         pred = self.layers[-1].numpy()
        
#         return np.sum(-np.log(np.abs(_labels.numpy()-pred))) / n_obs
        

    def cross_entropy(self, _x, _labels):
        m = _labels.shape[0]
        print("M IS:", m)
        p = self.softmax_activation(_x)
        print("p shape:", p.shape)
        log_likelihood = -torch.log(_labels)
        loss = torch.sum(log_likelihood) / m
        return loss 
    
    def cross_entropy_delta(self, _x , _labels):
        m = _labels.shape[0]
        grad = self.softmax_activation(_x)
        grad[:, :] -= 1
        grad = grad/m
        return grad
    
    ## function to calculate the derivative of activation
    def sigmoid_delta(self, x):
        return self.sigmoid_activation(x) * (1 - self.sigmoid_activation(x))

    def backprop_and_update(self, _data, _labels):
#         print("\n")
#         print("BACK PROP \n")
        self.weight_change.append(self.weights[0][0][0].item())
        loss = self.calculate_loss(_labels)
#         loss = self.cross_entropy(self.get_output(), _labels)
        self.losses.append(loss*100)
        _as = copy.deepcopy(self.layers)
        _as.insert(0,_data) # insert input into the layer vals
        del _as[-1]
    #     print("\n Layer vals \n", _layer_vals)
        # Update parameters
        new_weights = []
        new_biases = []

        dz = self.loss_derivative(_labels)
#         dz = self.cross_entropy_delta(self.get_output(), _labels)
        m = _labels.shape[0]
        for a, w, b in zip(reversed(_as), reversed(self.weights), reversed(self.biases)):
#             print("shapes:", a.t().shape, dz.shape)
            dw = (1/m)*torch.mm(a.t().float(), dz.float())
            db = (1/m)*torch.sum(dz)
#             print("DZ: \n", dz.shape)
#             print("W: \n", w.t().shape)
#             print("DW: \n", dw.shape)
#             print("DB: \n", db.shape)
#             print("delta: \n", self.sigmoid_delta(a).shape)
            prod = torch.mm(dz.float(), w.t())
#             print("product: \n",  prod.shape)
#             dz = torch.dot(prod, self.sigmoid_delta(a).t())
            dz = prod
            
            new_w = w - self.lr*dw
            
            #weight dropout
            new_w = new_w.numpy()
            new_w[np.abs(new_w)<0.05]=0
            new_w = torch.from_numpy(new_w)
            
            
            new_b = b - self.lr*db
            
            new_weights.append(new_w)
            new_biases.append(new_b)
            
#         print("\n WEIGHT COMPARISON")
#         for w in reversed(new_weights):
#             print("NEW WT SHAPE: \n", w.shape)
#         for w in self.weights:
#             print("OLD WEIGHT SHAPE: \n", w.shape)
    
        new_weights.reverse()
        new_biases.reverse()
        
#         print("BEFORE ASSIGNMENT")
#         for w in new_weights:
#             print("NEW WT SHAPE: \n", w.shape)
                        
        self.weights = new_weights
        self.biases = new_biases
        
#         print("AFTER ASSIGNMENT")
#         for w in self.weights:
#             print("NEW WEIGHT SHAPE: \n", w.shape)
        
    def batch(self, _batch_size, _data, _labels):
        # should be made more elegant, i'm aware this solution is trash
        d_all = np.append(_data.numpy(), _labels.numpy(), axis=1)
        np.random.shuffle(d_all)
        batch = torch.from_numpy(d_all[:_batch_size])
        
        return batch[:,:_data.shape[1]], batch[:,_data.shape[1]:]

    def train(self, _train_data, _train_labels, _n_epochs, _lr=0.01, batch_size=50, _test=False, _test_data=[], _test_labels=[]):
        self.lr = _lr
        for j in range(_n_epochs):
            if(batch_size==0):
                batch_data, batch_labels = _train_data, _train_labels
            else:
                batch_data, batch_labels = self.batch(batch_size, _train_data, _train_labels)
            self.forward_prop(batch_data)
            # test here 
            train_acc = self.get_accuracy(batch_data, batch_labels)
            self.train_accuracies.append(train_acc)
            
            if (j % 150 == 0):
                if(_test==True):
                    test_acc = self.get_accuracy(_test_data, _test_labels)
                    self.test_num.append(j)
                    self.test_accuracies.append(test_acc)
            if (j % 300 ==0):
                print("train accuracy at epoch", j, "is:", train_acc)
#                 print("weight shapes", self.weights[0].shape)
                if(_test==True):
                    print("test accuracy is:", test_acc)
            self.backprop_and_update(batch_data, batch_labels)
#             print("weight shapes after backprop", self.weights[0].shape)
        


        


        
    
    