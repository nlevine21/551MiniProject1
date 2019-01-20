from proj1_data_loading import loadData
from construct_dataset import buildMatricies
import numpy as np

# Variables selecting which bonus features to add
include_length = True
include_num_sentences = True

# First load the data using proj1_data_loading
data = loadData()

# Split the data into training, validation and test data according to specs
training_data = data[:10000]
validation_data = data[10000:11000]
test_data = data[11000:]

# Build the X and Y matricies for the three splits of data
X_training_data, Y_training_data = buildMatricies(training_data, include_length, include_num_sentences)
X_validation_data, Y_validation_data = buildMatricies(validation_data, include_length, include_num_sentences)
X_test_data, Y_test_data = buildMatricies(test_data, include_length, include_num_sentences)

print (X_test_data[998])
