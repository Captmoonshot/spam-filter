
import os
from preprocessing import make_bag, to_one_hot, oha_to_text, clean_line

ROOT_DIR 				= os.getcwd()
SPAM_DATA_PATH 			= os.path.join(ROOT_DIR, 'data', 'spam', 'spam.txt')
NOT_SPAM_DATA_PATH		= os.path.join(ROOT_DIR, 'data', 'spam', 'not-spam.txt')
SPAM_LABEL 				= 0
NOT_SPAM_LABEL 			= 1

MAX_LINES = 500

'''
with open('data/simple/pos.txt', 'r') as pos_f:
	lines = pos_f.readlines()
	for line in lines:
		escaped_line = line.replace('\n', '')
		bag = make_bag(escaped_line)


print(bag)
'''
bag = []

def file_to_bow(filepath=NOT_SPAM_DATA_PATH):
	global bag
	with open(filepath, 'r') as f:
		lines = f.readlines()[:MAX_LINES]
		for line in lines:
			escaped_line = clean_line(line)
			bag = make_bag(escaped_line)

file_to_bow(filepath=NOT_SPAM_DATA_PATH)
file_to_bow(filepath=SPAM_DATA_PATH)


# print("\nOur Bag-of-Words:\n {}\n".format(bag))

'''
Positive = label = 1
Negative = label= 0
'''

def file_to_oha(filepath=NOT_SPAM_DATA_PATH, label=1):
	my_oha = []
	labels = []

	#if filepath.endswith("neg.txt"):
		#label = 0
	with open(filepath, 'r') as f:
		lines = f.readlines()[:MAX_LINES]
		for line in lines:
			escaped_line = clean_line(line)
			oha = to_one_hot(escaped_line)
			labels.append(label)
			my_oha.append(oha)

	return my_oha, labels



pos_ohas, pos_labels = file_to_oha(filepath=NOT_SPAM_DATA_PATH, label=NOT_SPAM_LABEL)	# Label=1
neg_ohas, neg_labels = file_to_oha(filepath=SPAM_DATA_PATH, label=SPAM_LABEL)	# Label=0
"""
print("Positive One-Hot Array:\n {}".format(pos_ohas))
print("\nPositive Labels: {}".format(pos_labels))
print()
print()
print("Negative One-Hot Array:\n {}".format(neg_ohas))
print("\nNegative Labels: {}".format(neg_labels))
"""

# But we need both the pos_ohas and pos_labels put together
# We can use NumPy for this

import numpy as np 
X_pos = np.array(pos_ohas)
X_neg = np.array(neg_ohas)

X = np.concatenate((X_pos, X_neg), axis=0)
"""
print("\nOne Hot Array positive and negative together:\n", X)
print()
"""
# print(len(X))

# Now do the same with labels

y_pos = np.array(pos_labels)
y_neg = np.array(neg_labels)

y = np.concatenate((y_pos, y_neg), axis=0)

'''
X = training data
y = labels = target
'''
"""
print()
print(X)
print(y)
"""
# print(len(X) == 14)
# print(len(y) == len(X))

# Let's use scikit-learn to train our model
# We also need to shuffle the data because the first 7 observations are all 1s and
# The last 7 observations are all 0s

from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=0)
"""
print("\nShuffled Data:")
print()
print(X)
print()
print(y)
"""


from sklearn import svm

clf = svm.SVC()
clf.fit(X, y)

# Serializing the clf model with pickle
import pickle

pickle.dump(clf, open('data/pickles/classifier.pkl', 'wb'))

pickle.dump(bag, open('data/pickles/bow.pkl', 'wb'))

























