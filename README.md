# Machine Learning Algorithm
Machine learning algorithm created to classified the data read by a power meter. First a model is build (train and tested), then it is used to classified data. Beside of this algorith, a program is here to parse the data provided by the power reader so that it can be use with the machine learning algorithm.

## Prerequisites
Python 3 is needed for both programs.

### Data parser
No prerequisites.

### Machine learning Algorithm
The following libraries required to be install:
* scipy
* numpy
* matplotlib
* pandas
* sklearn
Use this command:
```
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

## Gettting Started

### Data parser
This program is creating/appending a csv file compatible with the machine learning program. 
The csv file is created based on the data received from the energy meter reader.
To execute this program, use Python 3 and the following inputs:
* Give as first parameter the data file
* Give as second parameter the machine from where the data come from if it is training data, otherwise put 0
* Give a third parameter for the name of the output file

Example:
```
python dataParser.py washingMachine.data washingMachine trainingData.csv
python dataParser.py washingMachine.data 0 data.csv
```

Output:
* CSV file

### Machine Learning Algorithm
Simply execute:
```
python machineLearning.py trainingData.csv data.csv
```