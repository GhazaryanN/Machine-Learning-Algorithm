"""
This program is creating/appending a csv file compatible with the machine learning program. 
The csv file is created based on the data received from the energy meter reader.
To execute this program, use Python 3
Input:
Give as first parameter the data file
Give as second parameter the machine from where the data come from if it is training data, otherwise put 0
Give a third parameter for the name of the output file

Output:
CSV file
"""

import sys

fileR = open(sys.argv[1])
fileW = open(sys.argv[3], "a")

# Function to check if a string can be a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Main start here

for line in fileR:
	if line != "\n":

		# Modify a little the lines to make the work easier
		line = line.replace("}", "")
		line = line.replace(",", "")

		# Split the lines by the space character
		line = line.split()


		# For each part of the splitted line, if it is a number, it is written in the csv file
		for l in line:
			if is_number(l):
				fileW.write(l + ",")

		fileW.write(sys.argv[2] + "\n")

fileW.close()
fileR.close()