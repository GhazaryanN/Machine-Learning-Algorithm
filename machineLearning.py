"""
The following libraries required to be install:
    scipy
    numpy
    matplotlib
    pandas
    sklearn
Use this command:
	python -m pip install --user numpy scipy scikit-learn matplotlib ipython jupyter pandas sympy nose

Input:
	Give as first parameter the training data file
	Give as second parameter the data to classify
	Optional: -v to have the details of the classification
"""

# Load libraries
import sys
import pandas
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataTrain", type=str, help="The data to train the model.")
parser.add_argument("data", type=str, help="The data to classify.")
parser.add_argument("-v", "--verbosity", action="count", default=0)
args = parser.parse_args()

# Check arguments number and validity
if len(sys.argv) < 3:
    raise TypeError("2 arguments are needed. The training data file, and the data to classify.")

try:
	open(args.dataTrain, 'r').readlines()
	open(args.data, 'r').readlines()
except FileNotFoundError:
	print("[FileNotFoundError] Wrong file or file path in one of the two given files.")
	sys.exit()

# Load training dataset
url = args.dataTrain
names = ['A', 'VA', 'Ea+', 'Bulk[ms]', 'W', 'Ms', 'CosP', 'VAR', 'VAC', 'Wh', 'class']
dataset = pandas.read_csv(url, names=names)

# descriptions of the columns (not the last one with the classes)
print("Training Data: ")
print(dataset.describe())
print()

# class distribution
print(dataset.groupby('class').size())
print()

# Split-out validation dataset
array = dataset.values
X = array[:,0:10]
Y = array[:,10]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
scoring = 'accuracy'

# Spot Check Algorithm
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(KNeighborsClassifier(), X_train, Y_train, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
print(msg)

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



# Classifications of the data
data = args.data
names = ['A', 'VA', 'Ea+', 'Bulk[ms]', 'W', 'Ms', 'CosP', 'VAR', 'VAC', 'Wh']
dataset = pandas.read_csv(data, names=names)
array = dataset.values
X = array[:,0:10]

# descriptions of the columns (not the last one with the classes)
print("Data: ")
print(dataset.describe())
print()

predictions = knn.predict(X)

predictions_d = {}

for i in range(len(X)):
	if args.verbosity:
		print("%s, Predicted=%s" % (X[i], predictions[i]))

	if not predictions[i] in predictions_d:
		predictions_d[predictions[i]] = 1
	else:
		predictions_d[predictions[i]] += 1


print("\nTotal of data to predict: " + str(len(X)))

print("\nList of predictions: ")
for name, value in predictions_d.items():
	print(str(value)+ " " + str(name) + " (" + str(value/len(X)*100) + "%)")