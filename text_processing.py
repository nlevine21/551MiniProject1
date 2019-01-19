# Preprocessing of the data to uncover the 160 most common words
#
# NOTE: this function takes ~4-5 minutes to run and it is not helpful
#       to run it each time when performing linear regression. Therefore,
#       it was run once and the output data can be found in words.txt
def preprocessText(data):
    # Obtain a list of all comments in lower case
    comment_list = list(map(lambda data_point: data_point["text"].lower(), data))

    # Divide each comment into words by splitting on blank space
    # Now we have a list where each element is a list of words that make up a comment
    word_list = list(map(lambda comment: comment.split(), comment_list))

    # Flatten our 2d list into a regular list of words 
    word_list = [word for comment in word_list for word in comment]

    # Obtain a list of all unique words
    unique_word_list = set(word_list)

    # Dictionary mapping each word to the number of occurences in the data set
    word_counts = {}

    for unique_word in unique_word_list:
        word_counts[unique_word] = word_list.count(unique_word)
    
    # Sort the most common words in descending order
    sorted_word_counts = sorted(word_counts.iteritems(), key=lambda (k,v): (v,k), reverse=True)
    
    # Only keep the 160 most common words
    sorted_word_counts = sorted_word_counts[:160]

    # Write the 160 most common words to words.txt
    with open('words.txt', 'w') as fout:
        for (word, count) in sorted_word_counts:
            fout.write('%s %d\n' % (word, count))
    fout.close()

# Function that returns the word count vector feature for a given comment
def getWordCountVector(comment):
    # Get the list of the 160 most common words by obtaining the data from words.txt
    word_list = []
    with open('words.txt', 'r') as fin:
        for i in range(160):
            word_list.append(next(fin).strip().split()[0])
    fin.close()

    # Build a list with each word in the comment
    comment = comment.lower()
    words_in_comment = comment.split()

    # Build a vector representing the counts for each of the 160 most common words in each comment
    word_count_vector = []
    for word in word_list:
        word_count_vector.append(words_in_comment.count(word))
    
    return word_count_vector
