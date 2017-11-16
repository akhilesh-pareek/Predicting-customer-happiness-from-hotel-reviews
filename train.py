import csv
import pickle
from random import shuffle
from preprocess import preprocess
from createBagOfWords import createBagOfWords
from featureExtraction import featureExtraction
from sklearn.linear_model import LogisticRegression

# load the training data from trainData.csv file
data = [];
fileObj = open("trainData.csv", encoding="utf-8");
csvReader = csv.reader(fileObj, delimiter=',');
for row in csvReader:
	data.append(row);

# first row is titles, so exclude it
data = data[1:];	

# randomly permute the data
shuffle(data);		# imported from package: random

reviews = [ x[1] for x in data ];
labels = [ x[2] for x in data ];
del data;

dataSize = len(reviews);
countHappy = sum([ 1 for x in labels if x == "happy"]);
print("The number of reviews with response 'happy':", countHappy);
print("The number of reviews with response 'not happy':", dataSize-countHappy);

# preprocess the review data
reviews = preprocess(reviews);
print("\nPreprocessing finished...\n");

# the split is 80 : 20 
# 20% is the testing data
# the rest of 80% is used for 3-fold cross validation, and finally the model is tested on remaining 20% data
# we perform 3-fold cross validation to find out the most suitable regularization parameter(hyper-parameter)
testSize = int(0.2*dataSize);
# select the last 20% of the data
# the final accuracy is calculated upon test set using the best selected model
testReviews = reviews[-testSize: ];	
testLabels = labels[-testSize: ];

#remaining data which will be used in 3-fold cross validation
reviews = reviews[: -testSize];
labels = labels[: -testSize];
dataSize = dataSize - testSize;


# create bag of words using best hyper-parameter value, and save with pickle
# bag of words is used for creating feature vector from review text
# we are choosing regularization parameter and the bag of words from k-fold cv

# start of 3-fold cross validation and grid search for hyper parameters

# list for k-fold sections. Using k = 3. Each tuple gives the section to be considered for cross validation
fold = [];
for i in range(3):
	start = int(i/3*dataSize);
	end = int((i+1)/3*dataSize);
	fold.append((start, end));

# grid for searching parameters
paramC = [ 0.01*(3**i) for i in range(2, 6)];
freqWord = [100*(2*i+1) for i in range(3, 10)];
grid = [];
for x in paramC:
	for y in freqWord:
		grid.append((x, y));
del paramC;
del freqWord;


# preform grid search using cv
bestParameter = None;
bestAccuracy = 0;
for g in grid:
	C = g[0];	# regularization parameter to be used in logistic regression
	freqWordSize = g[1];	# no of words to be taken from both frequent positive and negative words
	print("-----------------------------------------------");
	print("Parameters, ", "C=", C, "FreqWordSize=", freqWordSize);

	# perform kfold for each tuple in grid, k = 3 and obtain average accuracy
	avgAccuracy = 0;
	for i in range(len(fold)):
		# obtain the train and cv sets
		trainReviews = reviews[: fold[i][0]-1] + reviews[fold[i][1]: ];
		trainLabels = labels[: fold[i][0]-1] + labels[fold[i][1]: ];
		cvReviews = reviews[fold[i][0]: fold[i][1]-1];
		cvLabels = labels[fold[i][0]: fold[i][1]-1];

		# along with the reviews and labels, number of most frequent words taken are also required to create bag of words
		bag = createBagOfWords(trainReviews, trainLabels, freqWordSize);
		print("Fold ", i + 1);
		print("Size of bag ", len(bag));

		trainFeatures = featureExtraction(trainReviews, bag);
		clf = LogisticRegression(C=C);
		clf.fit(trainFeatures, trainLabels);
		cvFeatures = featureExtraction(cvReviews, bag);
		predictions = clf.predict(cvFeatures);
		accuracy = 0;
		for i in range(len(predictions)):
			accuracy += (predictions[i] == cvLabels[i]);
		accuracy = 100 * accuracy/len(predictions);
		avgAccuracy += accuracy;
		print("Accuracy =", accuracy);
		print("-------------------------------------------\n");

	avgAccuracy /= 3;
	if (avgAccuracy > bestAccuracy):
		bestParameter = g;
		bestAccuracy = avgAccuracy;

print("Best accuracy and parameters:")
print(bestAccuracy);
print(bestParameter);

# train a classifier based upon the best parameters on the complete data
# find the accuracy of test set

bag = createBagOfWords(reviews, labels, bestParameter[1]);
features = featureExtraction(reviews, bag);
clf = LogisticRegression(C=bestParameter[0]);
clf.fit(features, labels);

testFeatures = featureExtraction(testReviews, bag);
testPredictions = clf.predict(testFeatures);
testAccuracy = 0;
for i in range(len(testPredictions)):
	testAccuracy += (testPredictions[i] == testLabels[i]);
print("\nTest Set Accuracy =", testAccuracy/len(testPredictions) * 100);
print(testPredictions);

# save the trained classifier and the bag of words
print("Saving the classifier and the bag of Words using pickle");
file = open("ClassifierPickle", "wb");
pickle.dump(clf, file);
file.close();

file = open("BagOfWordsPickle", "wb");
pickle.dump(bag, file);
file.close();