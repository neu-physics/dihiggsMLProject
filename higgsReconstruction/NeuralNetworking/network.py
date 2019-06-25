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
    
    #     n_epochs	# number of epochs, inputted as train param
    #     lr = 0.01 # learning rate, train parameter
    #     train_data = []	# inputted as train param
    #     train_labels = []	# inputted as train param
    #     test_data = []	# inputted as train param
    #     test_labels = []	## inputted as train param

    
    
    # methods
    def __init__(self, _nodes):
        nodes = _nodes # set array of nodes
        num_layers = len(_nodes)
#         n_hidden = len(_nodes)-2
        
        # initialize weights 
        weights = []
        for i in range(len(_nodes)-1):
            w = torch.randn(_nodes[i], _nodes[i+1])
            weights.append(w)
        # initialize biases
        biases = []
        for i in range(len(nodes)-1):
            b = torch.randn(1, nodes[i+1])
            biases.append(b)   
            
    ## sigmoid activation function using pytorch
    def sigmoid_activation(z):
        return 1 / (1 + torch.exp(-z))    
    
    def forward_prop(_data, _nodes, _weights, _biases):
        layer_vals = []
        a = _data
        for i in range(len(_nodes)-2): # feed thru hidden layers
            z = torch.mm(a, _weights[i]) + _biases[i]
            a = sigmoid_activation(z)
            layer_vals.append(a)
        z_out = torch.mm(a, _weights[-1]) + _biases[-1]
        output = sigmoid_activation(z_out)
        layer_vals.append(output)
    #     print(output)
        return layer_vals
    
    # loss computation
    # loss = y - output
    def calculate_loss(_layer_vals):
        return y - _layer_vals[-1]
    
    ## function to calculate the derivative of activation
    def sigmoid_delta(x):
      return x * (1 - x)

    def backprop_and_update(_layer_vals, _weights, _biases, _data):
        loss = calculate_loss(_layer_vals)
        deltas = [] # from first hidden to output
        ds = [] # from output to first hidden
        # compute derivative of error terms
        for a in _layer_vals:
            deltas.append(sigmoid_delta(a))
        loss_d = loss
        for d, w in zip(reversed(deltas), reversed(_weights)):
            dd = loss * d
            loss_d = torch.mm(d, w.t())
            ds.append(dd)
        learning_rate = 0.1
        _as = copy.deepcopy(_layer_vals)
        _as.insert(0,x)
        del _as[-1]
    #     print("\n Layer vals \n", _layer_vals)
        new_weights = []
        new_biases = []
        print("lengths", len(ds), len(_weights), len(_as))
        for d, w, a, b in zip(ds, reversed(_weights), _as, reversed(_biases)):
            wt = torch.mm(a.t(),d) * learning_rate
            new_weights.insert(0, wt)

            bi = b + d.sum()*learning_rate
            new_biases.insert(0,bi)
        return new_weights,  new_biases  

    def train(self, _train_data, _train_labels, _n_epochs, _lr=0.01, _test=False, _test_data=[], _test_labels=[]):
        
    for i in range(_n_epochs):
        y_out = forward_prop(_data, _nodes, weights, biases)
        weights, biases = backprop_and_update(y_out, weights, biases, _data)


        
    
    