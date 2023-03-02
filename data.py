import numpy as np


# Function to group the variables
def group_vars(data):
    # Extract first 8 variables
    group1 = [row[0:8].tolist() for row in data]

    # Extract next 400 variables
    group2 = [row[8:408].tolist() for row in data]

    # Extract next 400 variables
    group3 = [row[408:808].tolist() for row in data]

    # Return all groups
    return [group1, group2, group3]


# Load data from file
filename = input("Enter file name: ")
with open(filename, 'r') as file:
    data = [[float(x) for x in line.split()] for line in file]

# Group the variables
groups = group_vars(data)

# Example usage: print the mean of the first group
print("Mean of group 1: ", np.mean(groups[0]))
