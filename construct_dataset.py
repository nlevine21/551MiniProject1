from text_processing import getWordCountVector
import numpy as np
import re

# This function takes in a list of raw JSON data and returns
# the X and Y matricies used in linear regression
def buildMatricies(raw_data, include_length, include_num_sentences):

    # Build Y first since it is just a vector containing popularity scores
    y_list = list(map(lambda data_point: data_point["popularity_score"], raw_data))
    
    # Next we will work towards build X

    # Helper function to be used when performing mapping for the 
    # X features
    def xMapper(data_point):

        # Build a training example of features in the following order
        # x0 = children
        # x1 = controversiality
        # x2 = is_root
        # x3..xn-1 = text features
        # xn = 1 (bias term)
        x_example_features = []
        
        # Add children, controversiality and is_root features
        x_example_features.append(data_point["children"])
        x_example_features.append(data_point["controversiality"])
        x_example_features.append(int(data_point["is_root"]))

        # Obtain word count vector
        word_count_vector = getWordCountVector(data_point["text"])

        # Place the counts for each word as an additional feature in the example
        for count in word_count_vector:
            x_example_features.append(count)
        
        # Add the length of the comment if requested
        if include_length:
            x_example_features.append(len(data_point["text"]))
        
        # Add the number of sentences if requested
        if include_num_sentences:
            # Get the list of sentences by separating by a sentence ending character
            # We will say that a sentence must end with a '.', '!' or '?'
            sentences = re.split("[.!?]", data_point["text"])

            # Filter out any empty strings or white spaced strings in the sentences list
            sentences = list(filter(lambda sentence: sentence.strip() != "", sentences))

            ## Add the number of sentences as a feature
            x_example_features.append(len(sentences))
            

        # Add bias term
        x_example_features.append(1)

        return x_example_features
    
    x_list = list(map(xMapper, raw_data))
    
    # We want X and Y to be represented as a numpy array instead of a python list to ease further computations
    X = np.asarray(x_list)
    Y = np.asarray(y_list)
    
    return X, Y
   
 