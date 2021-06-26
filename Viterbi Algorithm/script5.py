# Importing header files requierd for the functionality and also other files including algorithm and various others.
import time
import corpus_parser as parser
import n_gramer
import smoother
import pos_tagger
import nltk
import pandas as pd
import matplotlib as plt
import csv
import json

# Indicating the STOP, START and RARE SYMBOL. Keeping a rare_word_frequency.
START_SYMBOL="*"
STOP_SYMBOL="STOP"
RARE_SYMBOL="_RARE_"
RARE_WORD_MAX_FREQ=5
LOG_PROB_OF_ZERO=-1000

''' This function recieves a set of sentences with "WORD_TAG" tokens.
words returned are a list of sentences where every element is a list of the tags of particular sentence.
tags returned are a list where every element is a list of the tags of a particular sentence.
'''
def split_wordtags(train):
	word_sentences,tag_sentences=parser.split_wordtags(train,start_word=START_SYMBOL,stop_word=STOP_SYMBOL)
	words=word_sentences
	tags=tag_sentences

	return words,tags

'''This function calculates trigram probabilities. 
Returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probabilities.'''
def calc_trigrams(tags):
	ngrams,corpus_size,sentence_count=n_gramer.make_ngrams(tags,2,START_SYMBOL,STOP_SYMBOL)
	n_gramer.calculate_ngram_probabilities(ngrams,corpus_size,sentence_count)

	q_values=ngrams[1]
	return q_values

''' returns a set of words that occur more that 5 times in the file'''
def calc_known(words):
	known_words=smoother.words_over_n_set(RARE_WORD_MAX_FREQ,words)
	return known_words

''' replaces the words with less than RARE_WORD_MAX_FREQ with the word "_RARE_"'''
def replace_rare(words,known_words):
	words_rare=smoother.replace_rare_words(words,known_words,RARE_SYMBOL)
	return words_rare

''' e_values is a python dictionary which gives us the probabilty of a tag given word. 
	The taglist is set of all_possible tags for the data set.'''
def calc_emission(words_rare,tags):
	e_values,known_tags=pos_tagger.emission_probabilities_from(words_rare,tags)
	taglist=known_tags
	return e_values,taglist

''' This function trains up using the training data and returns the tagged sentences in the WORD_TAG method.'''
def viterbi(test_words,taglist,known_words,q_values,e_values):
	tagged=pos_tagger.tag(test_words,taglist,known_words,q_values,e_values)
	return tagged

''' The main function called '''
def main():
	
	infile=open("Brown_train.txt","r")
	file=infile.readlines()
	infile.close()

	# Get all tags in the file
	all_tags=get_all_tags(file)

	# Creating a dictionary of tags to store the confusion matrix.
	all_tags_dic={}
	for tag in all_tags:
		all_tags_dic[tag]={}

	for tag in all_tags:
		for tag2 in all_tags:
			all_tags_dic[tag][tag2]=0

	#print(all_tags_dic)
	fields=[tag for tag in all_tags]

	# First fold
	training_data=file[:int(len(file)*0.67)]
	testing_lines=file[(int(len(file)*0.67))+1:]
	all_tags_dic=folds(file,training_data,testing_lines,all_tags_dic,all_tags,1)
	write_file(all_tags_dic,1,fields)

	# Second fold
	training_data=file[:int(len(file)*0.34)]
	testing_lines=file[int(len(file)*0.34)+1:int(len(file)*0.67)+1]
	training_data.extend(file[int(len(file)*0.67)+1:])
	all_tags_dic=folds(file,training_data,testing_lines,all_tags_dic,all_tags,2)
	write_file(all_tags_dic,2,fields)

	# Third fold
	training_data=file[int(len(file)*0.34)+1:]
	testing_data=file[:int(len(file)*0.34)]
	all_tags_dic=folds(file,training_data,testing_lines,all_tags_dic,all_tags,3)
	write_file(all_tags_dic,3,fields)

	#print(all_tags_dic)

# Writes the confusion matrix to a csv file.
def write_file(all_tags_dic,fold,fields):

	fields=sorted(fields)
	fields=['tag']+fields
	string="Confusion_Matrix_Bigram"+str(fold)+".csv"
	with open(string,"w") as f:
		w = csv.DictWriter(f, fields )
		w.writeheader()
		for key,val in sorted(all_tags_dic.items()):
		    row = {'tag': key}
		    row.update(val)
		    w.writerow(row)

# Runs the algorithm for 1,2,3 folds
def folds(file,taining_data,testing_lines,all_tags_dic,all_tags,fold):

	# Taking in the start time
	start=time.time()
	training_data=file[:int(len(file)*0.67)]
	testing_lines=file[(int(len(file)*0.67))+1:]

	sentences=[]
	for sentence in testing_lines:
		sentence_words=[]
		word_tag_sentence=sentence.split()
		for word_tag in word_tag_sentence:
			delimiter_index=word_tag.find('_')
			word=word_tag[:delimiter_index]
			tag=word_tag[delimiter_index+1:]
			sentence_words.append(word)
		sentences.append(sentence_words)

	#print(sentences)
	test=sentences

	# Getting the list of words and tags
	words,tags=split_wordtags(training_data)

	# Getting the trigram probabilities
	q_values=calc_trigrams(tags)

	# Get words with frequency>5
	known_words=calc_known(words)

	# Replacing rare words
	words_rare=replace_rare(words,known_words)

	# Gets emission probabilities.
	e_values,taglist=calc_emission(words_rare,tags)

	# Gets the viterbi tagged sentences
	viterbi_tagged=viterbi(test,taglist,known_words,q_values,e_values)
	#print(len(viterbi_tagged))
	#print(len(test))
	#return viterbi_tagged

	# Counting the accuracy.
	count=0
	ans=0

	for i in range(len(testing_lines)):
		test_sentence=testing_lines[i]
		viterbi_sentence=viterbi_tagged[i]
		test_sentence_split=test_sentence.split()
		viterbi_sentence_split=viterbi_sentence.split()
		for j in range(len(test_sentence_split)):
			count+=1
			word_tag_test=test_sentence_split[j]
			delimiter_index=word_tag_test.find('_')
			word_test=word_tag_test[:delimiter_index]
			tag_test=word_tag_test[delimiter_index+1:]
			if (j<len(viterbi_sentence_split)):
				word_tag_viterbi=viterbi_sentence_split[j]
				delimiter_index=word_tag_viterbi.find('_')
				word_viterbi=word_tag_viterbi[:delimiter_index]
				tag_viterbi=word_tag_viterbi[delimiter_index+1:]			
				if (word_viterbi==word_test):
					if (tag_viterbi==tag_test):
						ans+=1
				else:
					count-=1
			else:
				count-=1

	accuracy=(ans*2/count)*100
	print("Accuracy = ",accuracy)
	#print("tags = ",tags)

	# Computing the confusion matrix
	for i in range(len(testing_lines)):
		test_sentence=testing_lines[i]
		viterbi_sentence=viterbi_tagged[i]
		test_sentence_split=test_sentence.split()
		viterbi_sentence_split=viterbi_sentence.split()
		for j in range(len(test_sentence_split)):
			word_tag_test=test_sentence_split[j]
			delimiter_index=word_tag_test.find('_')
			word_test=word_tag_test[:delimiter_index]
			tag_test=word_tag_test[delimiter_index+1:]
			if (j<len(viterbi_sentence_split)):
				word_tag_viterbi=viterbi_sentence_split[j]
				delimiter_index=word_tag_viterbi.find('_')
				word_viterbi=word_tag_viterbi[:delimiter_index]
				tag_viterbi=word_tag_viterbi[delimiter_index+1:]			
				if (tag_viterbi==tag_test) and (tag_test in all_tags) and word_test==word_viterbi:
					all_tags_dic[tag_test][tag_viterbi]+=1
				if (word_viterbi==word_test) and (tag_viterbi and tag_test):
					all_tags_dic[tag_viterbi][tag_test]+=1
					all_tags_dic[tag_test][tag_viterbi]+=1

	# Calculating precision, recall, f1 score.
	calculate_values(all_tags_dic,all_tags,fold)

	end=time.time()
	print("Time = ",end-start)
	return all_tags_dic

''' Function that helps in calculating the precision, recall and f1score'''
def calculate_values(all_tags_dic,all_tags,fold):
	tag_wise_precision={}
	tag_wise_recall={}
	tag_wise_f1score={}
	for tag in all_tags:
		false_positive=0
		false_negative=0
		true_positive=all_tags_dic[tag][tag]
		for other_tags in all_tags:
			if tag!=other_tags:
				false_positive+=all_tags_dic[other_tags][tag]
				false_negative+=all_tags_dic[tag][other_tags]
		
		if (true_positive+false_positive)==0:
			precision=0
		else:
			precision=true_positive/(true_positive+false_positive)
		if (true_positive+false_negative)==0:
			recall=0
		else:
			recall=true_positive/(true_positive+false_negative)
		if (recall==0 or precision==0):
			f1_score=-1000
		else:
			f1_score=2/((1/recall)+(1/precision))
		tag_wise_precision[tag]=precision
		tag_wise_recall[tag]=recall
		tag_wise_f1score[tag]=f1_score

	# Writing the precision, recall and f1score in a file
	string="OutputFold_"+str(fold)+"_Bigram.txt"
	with open(string,'w') as text_file:
		text_file.write(json.dumps(tag_wise_precision))
		text_file.write(json.dumps(tag_wise_recall))
		text_file.write(json.dumps(tag_wise_f1score))

# Function that gives us all possible tags
def get_all_tags(file):
	
	all_tags=set()
	for sentence in file:
		sentence_words=[]
		word_tag_sentence=sentence.split()
		for word_tag in word_tag_sentence:
			delimiter_index=word_tag.find('_')
			word=word_tag[:delimiter_index]
			tag=word_tag[delimiter_index+1:]
			all_tags.add(tag)

	return list(all_tags)

# Implementing the main function
main()