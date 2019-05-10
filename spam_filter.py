import sys
import pandas as pd
import math
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict

def concatenate_text(classes, document, c):
	text_in_c = []
	for i in range(len(document)):
		if c == classes[i]:
			text_in_c.extend(document[i])
	return text_in_c

class spam_filter():
        data = pd.read_csv('email.csv')
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        stops = stopwords.words('english')
        stemmer = PorterStemmer()
        feature = []
        label = []
        vocabulary = []
        prior = {}
        condprob = defaultdict(dict)
        classes = [0, 1]

        def __init__(self):
                data = self.data
                tokenizer = self.tokenizer
                stops = self.stops
                stemmer = self.stemmer
                feature = self.feature
                label = self.label
                vocabulary = self.vocabulary
                prior = self.prior
                condprob = self.condprob
                classes = self.classes


                for i in range(len(data)):
                        
                        tokens = tokenizer.tokenize(data['text'][i])
    
                        # Remove stop words
                        final_tokens = []
                        for token in tokens: 
                                token = token.lower()
                                if token not in stops:
                                        token = stemmer.stem(token)
                                        final_tokens.append(token) 
                                        if token not in vocabulary:
                                                vocabulary.append(token)
                        feature.append(final_tokens)
                        label.append(data['spam'][i])

                total_document = len(feature)
                total_term = len(vocabulary)

                for c in classes:
                        # Count how many documents are in class c
                        document_in_c = label.count(c)
                        prior[c] = document_in_c/float(total_document)

                        # Concatenate all the text of class c in one list
                        text_in_c = concatenate_text(label, feature, c)

                        for term in vocabulary:
                                # Count how many term t are in class c
                                Tct = text_in_c.count(term)
                                condprob[term][c] = (Tct + 1)/(len(text_in_c) + total_term)

        def classify(self, query):
                # Apply tokenize, stop words, and stemming to query
                query_vocab = []
                terms = self.tokenizer.tokenize(query)
                for term in terms:
                        term = term.lower()
                        if term not in self.stops:
                                term = self.stemmer.stem(term)
                                query_vocab.append(term) 

                score = {}
                for c in self.classes:
                        score[c] = math.log(self.prior[c])
                        for term in query_vocab:
                                if term in self.condprob:
                                        score[c] += math.log(self.condprob[term][c])

                return max(score, key=score.get)
