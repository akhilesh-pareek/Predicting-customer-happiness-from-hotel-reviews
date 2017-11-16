# remove stop words, punctuations and digits, and convert everything to lower case

# tokenize the review sentence by using space (' ') as seperator
# the sentence is converted to a list of words

def preprocess(data):

	# load stop words from the file "stop_words.txt"
	# stop words are the most common english words which only act as fillers in the sentences
	# removing stop words will reduce the noise and the dimension of the features
	# the list of stop words must not contain words which are specific to the dataset but are otherwise unnecessary
	stop_words = set();
	file = open("stop_words.txt", "r");
	for line in file:
		stop_words.add(line[:-1]);
	file.close();

	# load negations from the file "negations.txt"
	# negations are common occuring words which invert the meaning of words and sentences
	# e.g. not, isn't, wasn't, etc. which leads to difference between, "good" vs "not good", "okay" vs "isn't okay"
	# negations must not be included in stop words
	negations = set();
	file = open("negations.txt", "r");
	for line in file:
		negations.add(line[:-1]);
	file.close();

	# process each review individually
	# the string of review text is converted to a list of words(strings)
	for i in range(len(data)):
		temp = "";	# for building words
		processedReview = [];
		for ch in data[i]:
			# if character is alphabet append it to current word, i.e. temp
			if (ch.isalpha()):
				temp += ch.lower();		# convert to lower case

			# if there is a space ' ', append the word to the list processedReview
			# stop words are ignored
			elif (ch == ' '):
				if (temp != "" and temp not in stop_words):		# temp != "" is checked to take care of consecutive spaces 
					processedReview.append(temp);		
				temp = "";

		# append the last word
		if (temp != "" and temp not in stop_words):
			processedReview.append(temp);
		
		# process negations by combining negation with the word that follows it
		newList = [];
		j = 0;
		while (j < len(processedReview)-1):
			if (processedReview[j] in negations):
				newList.append(processedReview[j] + " " + processedReview[j+1]);
				j += 2;
			else:
				newList.append(processedReview[j]);
				j += 1;

		# replace old review with the processed list
		data[i] = newList;	

	return data;