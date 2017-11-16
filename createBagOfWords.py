def createBagOfWords(reviews, labels, freqWordSize):
	positiveWords = {};		
	negativeWords = {};
	for i in range(len(reviews)):
		for word in reviews[i]:
			if (labels[i] == "happy"):
				if (word not in positiveWords):
					positiveWords[word] = 0;
				positiveWords[word] += 1;
			else:
				if (word not in negativeWords):
					negativeWords[word] = 0;
				negativeWords[word] += 1;

	# list of frequent positive and negative words sorted in descending order based on number of occurence
	frequentPosWords = [ x[0] for x in sorted(positiveWords.items(), key=lambda x:-x[1]) ];
	frequentPosWords = set(frequentPosWords[: freqWordSize]);

	frequentNegWords = [ x[0] for x in sorted(negativeWords.items(), key=lambda x:-x[1]) ];
	frequentNegWords = set(frequentNegWords[: freqWordSize]);

	bagCounter = 0;		# use this for bag of words vectorization
	bag = {};
	for x in frequentPosWords:
		if x not in bag:
			bag[x] = bagCounter;
			bagCounter += 1;
			
	for x in frequentNegWords:
		if x not in bag:
			bag[x] = bagCounter;
			bagCounter += 1;

	return bag;