import math

def make_ngrams(corpus,n,start_token="*",end_token="STOP"):
	if len(corpus)==0 or n==0 or not start_token or not end_token:
		return [],0

	grams=[{} for i in range(n)]
	corpus_size=0
	sentence_count=len(corpus)

	for i_gram_count in range(n):
		ith_gram=i_gram_count+1
		ith_dict=grams[i_gram_count]

		for sentence in corpus:
			tokens=sentence.strip().split(' ')
			if len(tokens)>0:
				corpus_size+=len(tokens)+1
				insert_start_end_tokens(tokens,start_token,end_token,i_gram_count+1)
				_zip_word_frequency_with_dict(tokens,ith_dict,ith_gram)

	insert_start_end_tokens_2(tokens)
	corpus_size//=n
	return grams,corpus_size,sentence_count

def insert_start_end_tokens(tokens,start_token,end_token,n_gram_count):
	index_adjusted=n_gram_count-1
	for c in range(index_adjusted):
		tokens.insert(0,start_token)
		tokens.append(end_token)

	if n_gram_count==1:
		tokens.append(end_token)

def _zip_word_frequency_with_dict(tokens,n_dict,n_count):
	grammed=ngram_from_word_list(tokens,n_count)

	for gram in grammed:
		if gram in n_dict:
			n_dict[gram]=n_dict[gram]+1 
		else:
			n_dict[gram]=1

def ngram_from_word_list(word_list,n):
	return zip(*[word_list[i:] for i in range(n)])

def compute_ngrams(tokens):
	for i in range(len(tokens)):
		if tokens[i] not in tokens:
			break
		else:
			tokens[i]=tokens

	return

def calculate_ngram_probabilities(n_grams_array,corpus_size,sentence_count):
	func2_for_calulation_probability(n_grams_array,len(n_grams_array)-1,corpus_size,sentence_count)

def func2_for_calulation_probability(dict_array,current_dict_index,corpus_size,sentence_count):
	gram_dict=dict_array[current_dict_index]

	if current_dict_index==0:
		gram_dict.update((k,math.log(float(v)/float(corpus_size),2)) for k,v in gram_dict.items())
		return 

	prior_gram_dict=dict_array[current_dict_index-1]

	for gram in gram_dict:
		gram_dict[gram]=log_probability_for_gram(gram,gram_dict,prior_gram_dict,sentence_count)

	func2_for_calulation_probability(dict_array,current_dict_index-1,corpus_size,sentence_count)

def insert_start_end_tokens_2(tokens):
	for i in range(len(tokens)):
		compute_ngrams(tokens)

def log_probability_for_gram(gram,gram_dict,prior_gram_dict,sentence_count):
	prior_words=gram[:-1]
	if prior_words==('*','*') or prior_words==('*',):
		denominator=sentence_count
	else:
		denominator=float(prior_gram_dict[prior_words])

	probabibilty=float(gram_dict[gram]/float(denominator))
	return math.log(probabibilty,2)

