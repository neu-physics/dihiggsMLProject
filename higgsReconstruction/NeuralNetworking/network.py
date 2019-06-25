class Network: 
    
    # instance variables
    nodes = []	# an array of numbers of nodes
    n_layers = 0	# determined by lentgh of nodes array
    n_hidden = 0 # n_layers - 2
    weights = []	# array of each weight matrix, len(weights) = num_layers -1
    biases = []	# array of each bias vector, len(biases) = num_layers -1
    train_data = []	# inputted as train param
    train_labels = []	# inputted as train param
    test_data = []	# inputted as train param
    test_labels = []	## inputted as train param
    n_epochs	# number of epochs, inputted as train param
    losses = []	# initialized as empty array, updated through training
    train_accuracies = []	# initialized as empty array, updated through training
    test_accuracies = []	# initialized as empty array, updated through training
    test_num = []	# initialized as empty array, updated through training
    output = []	# initialized as empty array, updated through training
    loss = [] 	# current loss value, train_labels - output
    wasTested = False	# boolean which sais whether or not the network was tested while training, used for plotting
    
    
    # methods
    def __init__(self, _nodes):
        nodes = _nodes # set array of nodes
        num_layers = len(_nodes)
        n_hidden = len(_nodes)-2
        
        # initialize weights 
        
        # initialize biases
        
    def train(self, _train_data, _train_labels, _n_epochs, _test=False, _test_data=[], _test_labels=[]):
    
    