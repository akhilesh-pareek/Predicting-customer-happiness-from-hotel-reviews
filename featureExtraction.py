def featureExtraction(reviews, bag):
	# load negations from file "negations.txt"
	negations = set();
	file = open("negations.txt", "r");
	for line in file:
		negations.add(line[:-1]);
	file.close();

	# length of each feature vector is equal to length of bag
	features = [];
	for row in reviews:
		temp = [0 for x in range(len(bag))];
		for word in row:
			contribution = 1;
			# if word is negated then make its contribution equal to 0, i.e. it is not present with its actual meaning
			s = word.split(' ');
			if (len(s) == 2 and s[0] in negations):
				# print(s);
				contribution *= 0;

			# add its contribution to the bag of words
			if word in bag:
				temp[bag[word]] += contribution;

		features.append(temp);
	
	return features;
