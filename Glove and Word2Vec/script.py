# Importing the necessary header files required

import csv
import re
import numpy as numpy
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
from glove import Corpus,Glove

# Ignoring warnings
warnings.filterwarnings(action='ignore')

# Reading file data of BingLiu.csv

BingLiu_data=[]
with open('BingLiu.csv','r') as file:
	reader=csv.reader(file)
	for row in reader:
		BingLiu_data.append(row)

# Reading files hindi.txt and english.txt and converting applying word_tokenize and sent_tokenize so as to obtain proper lexicons.

sample=open('hindi.txt','r')
s=sample.read()

f=s.replace('\n'," ")
hindi_data=[]

for i in sent_tokenize(f):
	temp=[]

	for j in word_tokenize(i):
		temp.append(j.lower())

	hindi_data.append(temp)

sample=open('english.txt','r')
s=sample.read()

f=s.replace('\n'," ")
english_data=[]

for i in sent_tokenize(f):
	temp=[]

	for j in word_tokenize(i):
		temp.append(j.lower())

	english_data.append(temp)

# Reading the english hindi dictionary text file and creating a dictionary for fast and easy access

english_hindi_dictionary={}
with open('english-hindi-dictionary.txt') as file:
	for translation in file:
		translationsplit=translation.split()
		if len(translationsplit)<=2:
			english_hindi_dictionary[translationsplit[0]]=translationsplit[0]
		else:
			tomergeintoone=translationsplit[2:]
			hindi_translation=""
			for i in range(len(tomergeintoone)-1):
				hindi_word=tomergeintoone[i]
				hindi_translation+=hindi_word+" "
			hindi_translation+=tomergeintoone[-1]
			english_hindi_dictionary[translationsplit[0]]=hindi_translation

# Creating the L1_dictionary 

L1_dictionary={}
for row in BingLiu_data:
	index=row[0].index('\t')
	english_word=row[0][:index]
	polarity=row[0][index+1:]
	if english_word in english_hindi_dictionary.keys():
		hindi_word=english_hindi_dictionary[english_word]
		
		L1_dictionary[english_word]=[]
		L1_dictionary[english_word].append(hindi_word)
		L1_dictionary[english_word].append(polarity)

BingLiu_dictionary={}
for row in BingLiu_data:
	word_polarity=row[0].split()
	word,polarity=word_polarity[0],word_polarity[1]
	BingLiu_dictionary[word]=polarity

# Obtaining all the files that are created to a text file

#print(L1_dictionary)
#print(english_hindi_dictionary)
#print(hindi_data)
#print(english_data)
#print(BingLiu_dictionary)	

# The Word2Vec models using gensim

model1=gensim.models.Word2Vec(sentences=english_data,min_count=1)
#print(model1)

model2=gensim.models.Word2Vec(sentences=hindi_data,min_count=1)
#print(model2)

word_vectors1=model1.wv
word_vectors2=model2.wv
#vec_i=model1.wv["abound"]
#vec_2=model2.wv["(Hindi_word)"]

# The Glove model creation using glove_python module

'''corpus1=Corpus()
corpus1.fit(english_data,window=100)

model3=Glove(no_components=30,learning_rate=0.15)

model3.fit(corpus1.matrix,epochs=100,no_threads=100,verbose=True)
model3.add_dictionary(corpus1.dictionary)
model3.save('glove.model3')

#vec_puree=glove.word_vectors[glove.dictionary['puree']]

corpus2=Corpus()
corpus2.fit(hindi_data,window=100)

model4=Glove(no_components=30,learning_rate=0.02)

model4.fit(corpus2.matrix,epochs=100,no_threads=100,verbose=True)
model4.add_dictionary(corpus2.dictionary)
model4.save('glove.model4')

word_vectors3=model3.word_vectors
word_vectors4=model4.word_vectors
dictionary3=model3.dictionary
dictionary4=model4.dictionary'''

#print(model3)
#print(model4)
#model3=glove.load('glove.model3')
#model4=glove.load('glove.model4')
#print(word_vectors3)
#print(word_vectors4)
#print(len(dictionary3.keys()))
#print(len(dictionary4.keys()))

beg_len=len(L1_dictionary.keys())
cnt=0
prev_len=0
new_additions_dic={}

# Performing the steps 3,4 and 5 for Word2Vec

while True:
	words=L1_dictionary.keys()
	prev_len=len(words)
	new_dic={}
	#print(len(words))
	for word in words:
		word_content=L1_dictionary[word]	
		hindi_word=word_content[0]
		polarity=word_content[1]
		if word in word_vectors1 and word_content[0]!=word and hindi_word in word_vectors2:
			five_most_similar1=model1.most_similar(positive=[word],topn=5)
			#if word=="good":
			#	print(five_most_similar1)
			english_words_to_check=[]
			for i in range(5):
				english_words_to_check.append(five_most_similar1[i][0])
			five_most_similar2=model2.most_similar(positive=[hindi_word],topn=5)
			#if hindi_word=="अच्छा":
			#	print(five_most_similar2)
			hindi_words_to_check=[]
			for j in range(5):
				hindi_words_to_check.append(five_most_similar2[j][0])
			for i in range(5):
				for j in range(5):
					if english_words_to_check[i] in english_hindi_dictionary.keys():
						if english_hindi_dictionary[english_words_to_check[i]]==hindi_words_to_check[j]:
							new_dic[english_words_to_check[i]]=[]
							new_dic[english_words_to_check[i]].append(hindi_words_to_check[j])
							new_dic[english_words_to_check[i]].append(polarity)

	for key in new_dic.keys():
		L1_dictionary[key]=new_dic[key]
		new_additions_dic[key]=new_dic[key]		

	this_len=len(L1_dictionary.keys())
	if this_len==prev_len:
		break

# Obtaining outputs for Word2Vec

end_len=len(L1_dictionary.keys())
additions=end_len-beg_len
print("The number of addtions into the L1_dictionary = ",additions)
print()
print("The new additions in dictionary format = ")
print(new_additions_dic)
print("The new L1_dictionary dictionary = ")
print(L1_dictionary)

# Best results for Word2Vec

# additions_dic = {'one': ['एक', 'positive'], ',': [',', 'positive'], 'and': ['और', 'positive'], 'its': ['इसकी', 'positive'], '-': ['-', 'positive'], 'tablet': ['टैबलेट', 'positive'], 'on': ['पर', 'negative'], 'from': ['से', 'negative'], 'in': ['में', 'negative'], 'or': ['या', 'negative'], 'at': ['पर', 'negative']}
# additions = 11

# Performing steps 3,4 and 5 for Glove model

'''beg_len=len(L1_dictionary.keys())
cnt=0
prev_len=0
new_additions_dic={}
while True:

	words=L1_dictionary.keys()
	new_dic={}
	for word in words:
		word_content=L1_dictionary[word]	
		hindi_word=word_content[0]
		polarity=word_content[1]
		if word in dictionary3 and word_content[0]!=word and hindi_word in dictionary4:
			five_most_similar1=model3.most_similar(word,6)
			#if word=="good":
			#	print(five_most_similar1)
			english_words_to_check=[]
			for i in range(5):
				english_words_to_check.append(five_most_similar1[i][0])
			five_most_similar2=model4.most_similar(hindi_word,6)
			#if hindi_word=="अच्छा":
			#	print(five_most_similar2)
			hindi_words_to_check=[]
			for j in range(5):
				hindi_words_to_check.append(five_most_similar2[j][0])
			for i in range(5):
				for j in range(5):
					if english_words_to_check[i] in english_hindi_dictionary.keys():
						if english_hindi_dictionary[english_words_to_check[i]]==hindi_words_to_check[j]:
							new_dic[english_words_to_check[i]]=[]
							new_dic[english_words_to_check[i]].append(hindi_words_to_check[j])
							new_dic[english_words_to_check[i]].append(polarity)

	cnt+=len(new_dic.keys())
	for key in new_dic.keys():
		if key not in L1_dictionary.keys():
			L1_dictionary[key]=new_dic[key]
			new_additions_dic[key]=new_dic[key]

	this_len=len(L1_dictionary.keys())
	if this_len==prev_len:
		break
	prev_len=this_len

# Obtaining outputs for Glove model

end_len=len(L1_dictionary.keys())
additions=end_len-beg_len
print("The number of addtions into the L1_dictionary = ",additions)
print()
print("The new additions in dictionary format = ")
print(new_additions_dic)
print("The new L1_dictionary dictionary = ")
print(L1_dictionary)

# Best results for Glove model 

# additions_dic = {'fast': ['तेज', 'positive'], 'very': ['बहुत', 'positive'], 'nice': ['अच्छा', 'positive'], 'good': ['अच्छा', 'positive'], 'house': ['घर', 'positive'], 'small': ['छोटे', 'negative'], 'thing': ['बात', 'negative'], 'problem': ['समस्या', 'negative']}
# additions = 8'''
