import collections
import n_gramer

def replace_rare_words(corpus,known_words,rare_symbol):
	results=[]

	for sentence in corpus:
		dirty_words=sentence.strip().split()
		clean_words=[]

		for word in dirty_words:
			if word in known_words:
				clean_words.append(word)
			else:
				clean_words.append(rare_symbol)

		results.append(clean_words)

	return results

def words_over_n_set(n,corpus):
	
	results=set()
	freqs=frequency_dict(corpus)

	for k,v in freqs.items():
		if v>n:
			results.add(k)

	return results

def frequency_dict(corpus):

	all_words={}

	for sentence in corpus:
		word_list=sentence.strip().split(' ')
		for word in word_list:
			all_words[word]=all_words.get(word,0)+1

	return all_words

