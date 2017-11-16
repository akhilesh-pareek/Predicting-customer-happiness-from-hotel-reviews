import csv
import pickle
from preprocess import preprocess
from featureExtraction import featureExtraction
from sklearn.linear_model import LogisticRegression

# run this module only if train.py has been successfully executed atleast once

# load the classifier using pickle
# the classifier is pre-trained, we can directly use it to make predictions
file = open("ClassifierPickle", "rb");
clf = pickle.load(file);
file.close();
print("Classifier loaded...");

# load the bag of words
file = open("BagOfWordsPickle", "rb");
bag = pickle.load(file);
file.close();
print("\nBag of Words loaded...");

# load the prediction data from file, make classification and write to predictionResult.csv file
# all the data is read and written in csv format
data = [];

# since the csv file is encoded in utf-8, we need to explicity pass this information to open function
dataFile = open("predictionData.csv", encoding="utf-8");
csvDataReader = csv.reader(dataFile, delimiter=',');
for row in csvDataReader:
	data.append([row[0], row[1]]);		# i.e [User_ID, Review]
dataFile.close();
print("\nPrediction data loaded...");

# first row is titles
data = data[1:];

# preprocess the reviews for feature extraction
reviews = [ x[1] for x in data ];
reviews = preprocess(reviews);
print("\nPreprocessing finished...");

# extract features from the reviews
# the bag of words is already loaded from BagOfWordsPickle
features = featureExtraction(reviews, bag);		# returns a list of features vectors

# classifier is already loaded from ClassifierPickle
# make predictions using the classifier
# the output is a list : [response]
predictions = clf.predict(features);

# use newline='' to avoid extra newline at the end of row
resultFile = open("predictionResult.csv", "w", encoding="utf-8", newline='');	
csvResultWriter = csv.writer(resultFile, delimiter=',');

# heading: User_ID, Is_Response
csvResultWriter.writerow(["User_ID", "Is_Response"]);

# write prediction results
for i in range(len(predictions)):
	# save as [User_ID, Is_Response]
	csvResultWriter.writerow([data[i][0], predictions[i]]);
resultFile.close();
print("\nPredictions saved to file ...");