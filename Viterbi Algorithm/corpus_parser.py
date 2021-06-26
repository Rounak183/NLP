import n_gramer

def split_wordtags(corpus,delimiter='_',start_word='*',stop_word='STOP',ngram_used=2):
	tag_sentences=[]
	word_sentences=[]

	for sentence in corpus:
		word_list=sentence.strip().split(' ')
		words=[]
		tags=[]
		for el in word_list:
			delimiter_index=el.find('_')
			word=el[:delimiter_index]
			tag=el[delimiter_index+1:]
			words.append(word)
			tags.append(tag)

		n_gramer.insert_start_end_tokens(words,start_word,stop_word,ngram_used)
		n_gramer.insert_start_end_tokens(tags,start_word,stop_word,ngram_used)

		words_sentence=' '.join(words)
		tags_sentence=' '.join(tags)

		tag_sentences.append(tags_sentence)
		word_sentences.append(words_sentence)

	return word_sentences,tag_sentences