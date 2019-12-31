from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from pandas import read_csv
from sklearn import preprocessing
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import random
import matplotlib.pyplot as plt

#Some hyperparameters
testing_fraction = 0.3
dropout = 0.2
top10vars = ['deltaR(h1, h2)', 'deltaR(h1 jets)', 'deltaR(h2 jets)', 'hh_mass', 'h1_mass', 'h2_mass','hh_pt', 'h1_pt', 'h2_pt', 'scalarHT']

#1) Import dataset
    #Grab spreadsheets
    #Read spreadsheets through pandas
#2) Preprocess data
    #Sort into top 10 variables
    #Normalize data
    #Insert binary column into each df
    #Append qcd df to higgs df
    #Seperate dataset into inputs and outputs
    #Split data into training, validation, and testing sets
#3) Model building
    #Create model
    #Compile model
    #Train model
#4) Analyze model
    #Accuracy
    #Confusion matrix
    #ROC curve

#1) Import dataset
    #Grab spreadsheets
dihiggs_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\higgs\equalDijetMass.csv" #var1=h1_mass, var2=deltaR(h1,h2)
qcd_file = r"C:\Users\Colby\Desktop\Neu-work\delphes stuff\spreadsheets\qcd\equalDijetMass.csv"
    #Read spreadsheets through pandas
dfhiggs = read_csv(dihiggs_file)
dfqcd = read_csv(qcd_file)

#2) Preprocess data
    #Sort top 10 variables
higgstop10 = dfhiggs[top10vars]
qcdtop10 = dfqcd[top10vars]
    #Normalize data
normal_higgs = preprocessing.scale(higgstop10)
normal_higgs_df = DataFrame(normal_higgs, columns=higgstop10.columns)
normal_qcd = preprocessing.scale(qcdtop10)
normal_qcd_df = DataFrame(normal_qcd, columns=qcdtop10.columns)
    #Insert binary column into each df
normal_higgs_df['Result'] = 1
normal_qcd_df['Result'] = 0
print(normal_higgs_df.head(10))
print(normal_qcd_df.head(10))
    #Append qcd df to higgs df
dataset = normal_higgs_df.append(normal_qcd_df)

X = dataset.loc[:, dataset.columns != 'Result']
y = dataset.loc[:, 'Result']

#Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_fraction) #80% train, 20% test
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=testing_fraction) #80% train, 20% validation

#3) Model building
    #Create model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=10))
model.add(Dropout(dropout))
model.add(BatchNormalization())

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
    #Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    #Train model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2)

#4) Analyze model
    #Accuracy
trainscores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (trainscores[1]*100))
testscores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (testscores[1]*100))
    #Confusion matrix
y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True,
                 xticklabels=['QCD', 'Dihiggs'],
                 yticklabels=['QCD', 'Dihiggs'],
                 cbar=False, cmap='Blues')
ax.set_xlabel('Prediction')
ax.set_ylabel('Actual')
plt.show()
    #ROC curve
y_test_pred_probs = model.predict(X_test)
false_pos_rate, true_pos_rate, _ = roc_curve(y_test, y_test_pred_probs)
auc_keras = auc(false_pos_rate, true_pos_rate)
plt.plot(false_pos_rate, true_pos_rate, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot([0,1],[0,1], '--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

