from collections import Counter 
from re import sub, compile
import re
import tokenize
import matplotlib.pyplot as plt
import numpy as np

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.cutoff_freq = 40
		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)
		


	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """ 

		return re.sub(r"[^a-zA-Z\d\s]", '', text).lower().split()

		# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		# raise UnimplementedFunctionError("You have not yet implemented tokenize.")



	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self,corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """ 

		word2idx = {}
		idx2word = {}
		freq = {}
		index = 1
		for text in corpus:
			for token in self.tokenize(text):
				if token not in freq.keys():
					freq[token] = 1
				else:
					freq[token] += 1
					if freq[token] >= self.cutoff_freq:
						if token not in word2idx.keys():
							word2idx[token] = index
							idx2word[index] = token

							index += 1

		word2idx["UNK"] = index
		idx2word[index] = "UNK"

		return word2idx, idx2word, freq
	
		# REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		# raise UnimplementedFunctionError("You have not yet implemented build_vocab.")


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """ 

		x_axis = []
		y_token_freq = []
		y_cumulative = []
		sum_of_tokens = 0
		x_cutoff = 0
		y_cutoff = 0

		frequ = dict(sorted(self.freq.items(), key=lambda item: item[1], reverse=True))

		for key in frequ:
			sum_of_tokens += self.freq[key]
			y_token_freq.append(self.freq[key])
			x_axis.append(len(y_token_freq))

			if self.freq[key] >= self.cutoff_freq:
				x_cutoff += 1

		sum_of_tokens2 = 0
		i = 0
		for key in y_token_freq:
			sum_of_tokens2 += key
			y_cumulative.append(sum_of_tokens2/sum_of_tokens)

			i += 1
			if i == x_cutoff:
				y_cutoff = sum_of_tokens2/sum_of_tokens

		#Token Frequency Chart
		plt.figure()
		plt.plot(x_axis, y_token_freq, linestyle='-')		
		plt.xlabel('Token ID (sorted by frequency)')
		plt.ylabel('Frequency')
		plt.yscale('log')
		plt.title('Token Frequency Distribution')
		plt.hlines(self.cutoff_freq, xmin=0, xmax=len(y_token_freq), colors='red')
		plt.text(len(y_token_freq), self.cutoff_freq*0.7, f'freq={self.cutoff_freq}', ha='right', va='center', color='red')
		plt.savefig("token_frequency_chart.png")
		# plt.show()

		#Cumulative Fraction Covered
		plt.figure()
		plt.plot(x_axis, y_cumulative, linestyle='-')		
		plt.xlabel('Token ID (sorted by frequency)')
		plt.ylabel('Fraction of Token Occurrences Covered')
		plt.title('Cumulative Fraction Covered')
		plt.vlines(x_cutoff, ymin=0, ymax=1, colors='red')
		plt.text(x_cutoff, y_cutoff, f"{y_cutoff:1.2f}", ha='right', va='center', color='red')
		plt.savefig("cumulative_coverage_chart.png")
		# plt.show()
	
	    # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
		# raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")

