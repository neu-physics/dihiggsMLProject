# Package imports
import numpy as np
from sklearn.model_selection import train_test_split

# Matplotlib is a matlab like plotting library
import matplotlib
from matplotlib import pyplot as plt
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating |datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from scipy.stats import gaussian_kde

# Global Variables
losses = []
train_accuracies = []
test_accuracies = []
test_num = []

gradsdW1 = []
gradsdb1 = []
gradsdW2 = []
gradsdb2 = []
gradsdW3 = []
gradsdb3 = []

w00s = []
w01s = []
w02s = []
w03s = []
w04s = []

w10s = []
w11s = []
w12s = []
w13s = []
w14s = []

b0s = []
b1s = []
b2s = []
b3s = []
b4s = []

# Now we define all our funpnctions
def softmax(z):
#     z = np.array(z, dtype=np.float128)
    #Calculate exponent term first
#     exp_scores = np.exp(-1*z)
# #     print("IS THIS ZERO???", np.sum(exp_scores, axis=1, keepdims=True))
#     return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
      return 1 / (1 + np.exp(-z))




def relu(z): 
    return z 

# loss functions
def softmax_loss(y,y_hat):
    # Clipping value
    minval = 0.000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
    #loss = -1/m * np.sum(y * np.log(y_hat))
    return loss

def crossEntropy_loss(y, y_hat):
    m = y.shape[0]
    if y.all() == 1:
        return -1/m * np.sum(np.log(y_hat))
    else:
        return -1/m * np.sum(np.log(1 - y_hat))

def mse_loss(y, y_hat):
    m = y.shape[0]
    return np.sum((y_hat - y)**2) / m
    
def loss_derivative(y,y_hat):
    return (y_hat-y)

def tanh_derivative(x):
#     print("tanh_derivative val :", x)
    return (1 - np.power(1/x, 2))
    

# This is the forward propagation function
def forward_prop(model,a0):
    
    #Start Forward Propagation
    
    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'],model['b3']
    
    # Do the first Linear step 
    # Z1 is the input layer x times the dot product of the weights + our bias b
    z1 = a0.dot(W1) + b1
    
    # Put it through the first activation function
    a1 = relu(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Second activation function
    a2 = relu(z2)
    
    #Third linear step
    z3 = a2.dot(W3) + b3
    
    #For the Third linear activation function we use the softmax function, either the sigmoid of softmax should be used for the last layer
    a3 = relu(z3)
    
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'a3':a3,'z3':z3}
    return cache

# This is the BACKWARD PROPAGATION function
def backward_prop(model,cache,y):

    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3']
    # Load forward propagation results
    a0,a1, a2,a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
#     print(a3,"a3")
    # Get number of samples
    m = y.shape[0]
#     print("y shape is:", y.shape)
#     print(y[:4])
#     print("y_hat shape is:", y_hat.shape)
#     print(y_hat[:4])
    
    # Calculate loss derivative with respect to output
    dz3 = loss_derivative(y=y,y_hat=a3)
#     dz3 = 1.02

    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/m*(a2.T).dot(dz3) #dW2 = 1/m*(a1.T).dot(dz2) 
    
    # Calculate loss derivative with respect to second layer bias
    db3 = 1/m*np.sum(dz3, axis=0) 
    
    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T) ,tanh_derivative(a2)) 
    
    # Calculate loss derivative with respect to first layer weights
    dW2 = 1/m*np.dot(a1.T, dz2) 
    # Calculate loss derivative with respect to first layer bias
    db2 = 1/m*np.sum(dz2, axis=0) 
    
    dz1 = np.multiply(dz2.dot(W2.T),tanh_derivative(a1)) 
    
    dW1 = 1/m*np.dot(a0.T,dz1) 
    
    db1 = 1/m*np.sum(dz1,axis=0)
    
    # Store gradients
    grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads

#TRAINING PHASE
def initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim):
    
    losses = []
    train_accuracies = []
    test_accuracies = []
    test_num = []
    
    gradsdW1 = []
    gradsdb1 = []
    gradsdW2 = []
    gradsdb2 = []
    gradsdW3 = []
    gradsdb3 = []
    
    w00s = []
    w01s = []
    w02s = []
    w03s = []
    w04s = []

    w10s = []
    w11s = []
    w12s = []
    w13s = []
    w14s = []

    b0s = []
    b1s = []
    b2s = []
    b3s = []
    b4s = []
    
    # First layer weights
    W1 = 5 *np.random.randn(nn_input_dim, nn_hdim) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim))
    b1 = b1
    
    # Second layer weights
    W2 = 3 * np.random.randn(nn_hdim, nn_hdim) - 1
    
    # Second layer bias
    b2 = np.zeros((1, nn_hdim))
    b2 = b2
    W3 = 2 * np.random.rand(nn_hdim, nn_output_dim) - 1
    b3 = np.zeros((1,nn_output_dim)) 
    b3 = b3
    
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3':W3,'b3':b3}
    return model

def update_parameters(model,grads,learning_rate):
    # Load parameters
    W1, b1, W2, b2,b3,W3 = model['W1'], model['b1'], model['W2'], model['b2'],model['b3'],model["W3"]
    
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    W3 -= learning_rate * grads['dW3']
    b3 -= learning_rate * grads['db3']
    
    
    # load parameters into running lists
    w00s.append(W1[0][0]) # modifies global list
    w01s.append(W1[0][1]) # modifies global list
#     w02s.append(W1[0][2]) # modifies global list
#     w03s.append(W1[0][3]) # modifies global list
#     w04s.append(W1[0][4]) # modifies global list
    
    w10s.append(W1[1][0]) # modifies global list
    w11s.append(W1[1][1]) # modifies global list
#     w12s.append(W1[1][2]) # modifies global list
#     w13s.append(W1[1][3]) # modifies global list
#     w14s.append(W1[1][4]) # modifies global list
    
#     b0s.append(b1[0][0]) # modifies global list
#     b1s.append(b1[0][1]) # modifies global list
#     b2s.append(b1[0][2]) # modifies global list
#     b3s.append(b1[0][3]) # modifies global list
#     b4s.append(b1[0][4]) # modifies global list
    
    gradsdW1.append(grads['dW1'])
    gradsdb1.append(grads['db1'])
    gradsdW2.append(grads['dW2'])
    gradsdb2.append(grads['db2'])
    gradsdW3.append(grads['dW3'])
    gradsdb3.append(grads['db3'])
    

    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3':W3,'b3':b3}
    return model
def predict(model, x):
    # Do forward pass
    c = forward_prop(model,x)
    #get y_hat
    y_hat = c['a3']
    # plotArr.append([x, y_hat]) #added to make plot
    return y_hat
def calc_accuracy(model,x,y):
    # Get total number of examples
    m = y.shape[0]
    # Do a prediction with the model
    pred = predict(model,x)
    # Ensure prediction and truth vector y have the same shape
    pred = pred.reshape(y.shape)
    # Calculate the number of wrong examples
    error = np.sum(np.abs(pred-y))
    # Calculate accuracy
    return (m - error)/m * 100
def trainThenTest(model,X_,y_, test_data, test_labels, learning_rate=0.01, epochs=2001, print_loss=False):
    
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):

        # Forward propagation
        cache = forward_prop(model,X_)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        
#         print("in trainThenTest, y_ shape:", y_.shape)
#         print(y_[:4])
        grads = backward_prop(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
        # it is at this point in the training that the weights get added to the lists
    
        a3 = cache['a3']
        thisLoss = mse_loss(y_,a3) # set loss function here
        losses.append(thisLoss) # modifies global list
        y_hat = predict(model,X_) # getting rid of this because it's wrong
        y_true = y_.argmax(axis=1)
        accur = accuracy_score(a3,y_)
        train_accuracies.append(accur) # modifies global list
        
        if i % 50 == 0:
            placeholderVar = accuracy_score(a3, y_)
            test_accuracy = accuracyOfModel(model, test_data, test_labels)
            test_accuracies.append(test_accuracy) # modifies global list
            test_num.append(i)
        #Printing loss & accuracy every 100 iterations
        if print_loss and i % 300==0:
            print('Loss after iteration',i,':',thisLoss)
            print('Train Accuracy after iteration',i,':',accur*100,'%')
            print('Test Accuracy after iteration',i,':',test_accuracy*100,'%')
    return model

# TESTING PHASE
# test the accuracy of any model
def accuracyOfModel(_model, _testData, _testLabels):
    y_pred = predict(_model,_testData) # make predictions on test data
    y_true = _testLabels # get usable info from labels
    return accuracy_score(y_pred, y_true)

def accuracy_score(_outputNodes, _labels):
    for i in range(len(_outputNodes)-1):
        if _outputNodes[i][0]>.5:
            _outputNodes[i]=[1,0]
        else:
            _outputNodes[i]=[0,1]
    numWrong = np.count_nonzero(np.subtract(_outputNodes,_labels))/2
    return (len(_outputNodes)-numWrong)/len(_outputNodes)

    
def plotAccPerEpoch(title):
    print("Loss length:", len(losses))
    print("Train Accuracy length:", len(train_accuracies))
    print("test num len", len(test_num))
    plt.plot(losses, label="Loss")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.scatter(test_num, test_accuracies, label="Test Accuracy", s=16, color="green")
    #plt.plot(test_accuracies, label="Test Accuracy")
    plt.plot()
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.show()
    plt.clf()
    
def plotSomeWeights():
      
    plt.plot((w00s), label="Weight 0.0")
    plt.plot((w01s), label="Weight 0.1")
#     plt.plot((w02s), label="Weight 0.2")
#     plt.plot((w03s), label="Weight 0.3")
#     plt.plot((w04s), label="Weight 0.4")
    # plt.plot((w00s-(w00s[0])), label="Weight 0.0")
    # plt.plot((w01s-(w01s[0])), label="Weight 0.1")
    # plt.plot((w02s-(w02s[0])), label="Weight 0.2")
    # plt.plot((w03s-(w03s[0])), label="Weight 0.3")
    # plt.plot((w04s-(w04s[0])), label="Weight 0.4")
    plt.ylabel("Value")
    plt.xlabel("Epoch")
    plt.title("Weights From the First Input Node at Each Epoch")
    plt.legend()
    print("initial w00", w00s[0])
    print("final w00:",w00s[len(w00s)-1])
    print("change:", w00s[len(w00s)-1]- w00s[0])
    print("initial w01", w01s[0])
    print("final w01:",w01s[len(w01s)-1])
    print("change:", w01s[len(w01s)-1]- w01s[0])
#     print("initial w02", w02s[0])
#     print("final w02:",w02s[len(w02s)-1])
#     print("change:", w02s[len(w02s)-1]- w02s[0])
#     print("initial w03", w03s[0])
#     print("final w03:",w03s[len(w03s)-1])
#     print("change:", w03s[len(w03s)-1]- w03s[0])
#     print("initial w04", w04s[0])
#     print("final w04:",w04s[len(w04s)-1])
#     print("change:", w04s[len(w03s)-1]- w04s[0])
    plt.show()
    
def plotGrads():
#     print("gradsW1.shape:", gradsdW1.shape)
    print("beginning")
    print(gradsdW1[0])
    print("end")
    print(gradsdW1[len(gradsdW1)-2])
    
#     plt.plot(gradsdW1)
def plotBias():
    bias0 = np.array(b0s)
    plt.plot(bias0)