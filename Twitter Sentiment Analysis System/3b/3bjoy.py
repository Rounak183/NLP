#!/usr/bin/env python
# coding: utf-8

# ### In this program we predict stats using ngrams, Using feature, ngrams + 25 features seperately and all three models are predicted and applied on SVM, MLP and decision tree. So there are total 9 models. Program only take 30 seconds to predict fit all models without using any pickle form which shows that model is very very fast. 
# 
# 
# FEATURES
# 
# + col1 Elongated words
# + col2 number of hshtags (FOR BONUS)
# + col3 number of capitals letter
# + col4 number of tagged persons in a sentence (FOR BONUS)
# + col5 for negations
# + col6 count words having particular emotion(joy)
# + col7 aggregate hashtag emotion value
# + col8 Aggregate emotion score (Hashtags)
# + col9  Emoticons score:
# + col 10,11 Aggregate polarity scores:
# + col 12 13 Aggregate polarity scores (Hashtags)
# + col 14 15 16 17  Lexicon based Features:
# + col 18 19 Punctuation feature (BONUS)
# + col 20 negation feature (take whole phrase) (BONUS)
# + col 21 22 23 24 Vader
# + col25 Only hashtags predicting nature for sentence  (only for this training data) (FOR BONUS) 
# 
# 
# 
# ++ NGRAM (1,2) Features 

# IMPORTING LIBRARIES 

# In[1]:


import pandas
from time import time
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from os import path
import re
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sys import getsizeof
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier , MLPRegressor
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from sklearn.tree import DecisionTreeRegressor 


# Function to predict model where:
# regressor- Regressor used (SVC,MLP,Decision tree)
# traindata- training data 
# testdata- testing data

# In[2]:


def modelprid(regressor,traindata,testdata,trainres,testres): 
    start_time = time()
    model = regressor
    model.fit(traindata,trainres)
    sss = model.predict(testdata)
    #print("Accuracy:",metrics.accuracy_score(rte, predic))
    statis(testres,sss)

    #print("\nTime = ",time() - start_time)


# Following are the features used to predict model

# In[3]:


def enlongated(st):        # For enlongated words
	temp = re.sub(r'(.)\1+', r'\1\1', st)
	if len(temp) == len(st):
		return 0
	else:
		return 1
    
def hashtag(st):         # to count hashtags
    if st[0] == "#":
        return 1
    return 0 

def allcaps(st):        # to count all capital letters
    if st.isupper():
        return 1
    return 0

def taggedp(st):      # to count tagged persons in the sentence
    if st[0] == "@":
        return 1
    return 0

def negations(st):    # to count negations
    if st.lower() in ["not","nor","neither","no","never","nope"]:
        return 1
    return 0

def hashemo(st,dic):    # to count Aggregate emotion score
    tem = []
    for i in st:
        if i.lower() in dic:
            tem.append(dic[i.lower()])
    if tem == []:
        return 0
    return sum(tem)/len(tem)

def wordemo(st,dic):         # Counts Emotion word count:
    if st.lower() in dic:
        return 1
    return 0

def sentiment_scores(sentence):      # Used in vader
    sid_obj = SentimentIntensityAnalyzer() 
    return sid_obj.polarity_scores(sentence)
    #print(sentiment_dict)
            
def vader(score,trainall):   # Vader returns 4 feature (positivity, neutrality, negativity, compound)
    for i in range(len(trainall)):
        temp = sentiment_scores(trainall[i])
        #print(temp)
        score[i].append(temp["pos"])
        score[i].append(temp["neu"])
        score[i].append(temp["neg"])
        score[i].append(temp["compound"])
        
def emotionsfeat(st,dic):     #Aggregate emotion score (Hashtags)
    tem = []
    for i in st:
        if i.lower() in dic:
            tem.append(float(dic[i.lower()]))
    if tem == []:
        return 0
    return sum(tem)/len(tem)

def emoticonfeat(st,dic):  #  Emoticons score
    aa = 0
    for i in st:
        if i in dic:
            aa += dic[i]
    return aa

def sent140feat(st,dic):   #Aggregate polarity scores:
    posi = []
    neg = []
    for i in st:
        if i in dic:
            posi.append(dic[i][0])
            neg.append(dic[i][1])
    if posi == []:
        return 0,0
    return sum(posi)/len(posi) , sum(neg)/len(neg)

def hashsentfeat(st,dic):       #Aggregate polarity scores (Hashtags)
    posi = []
    neg = []
    for i in st:
        if i in dic:
            posi.append(dic[i][0])
            neg.append(dic[i][1])
    if posi == []:
        return 0,0
    return sum(posi)/len(posi) , sum(neg)/len(neg)

def mpqafeat(st,dic):            #Lexicon based Features
    posi = 0
    neg = 0
    for i in st:
        if i.lower() in dic:
            if dic[i.lower()] == "positive":
                posi += 1
            elif dic[i.lower()] == "negative":
                neg += 1
    return posi, neg

def bingliufeat(st,dic):         #Lexicon based Features
    posi = 0
    neg = 0
    for i in st:
        if i.lower() in dic:
            #print(i.lower())
            #print(dic[i.lower()])
            if dic[i.lower()] == "positive":
                posi += 1
            elif dic[i.lower()] == "negative":
                neg += 1
    return posi, neg

def puncfeat(i):              # Punctuation Feature
    if i[-1] == "!" or i[-1]=="?":
        return len(re.findall("[!?]{2,}",i)) , 1
    else:
        return len(re.findall("[!?]{2,}",i)) , 0
    
def negatfeat(i):           # Negation for whole phrase
    return len(re.findall(r"(?i)(\b(no|none|not|nothing|neither|never|can't|isn't|doesn't|wouldn't|shouldn't)\b).*(,|\.|:|;|!|\?)",i))    


# In[4]:


hashem = open("joyhashtag.txt","r")   
hashdic = {}   # dictionary to keep hashtag emotions
for i in hashem:
    tem = list(map(str,i.split()))
    hashdic[tem[1]] = float(tem[2])
    
wordem = open("wordemotion.txt","r")
wordemodic = {}      # count words for particular emotion (anger or joy)
for i in wordem:
    tem = list(map(str,i.split()))
    if tem[1] == "joy" and tem[2] == "1":
        wordemodic[tem[0]] = 1
        
emotions = pandas.read_csv("emotion.csv",sep = "\t",error_bad_lines=False,)  
#print(emotions)  # dictionary for emoticons aggregate
emotiondic = {}
for i in range(len(emotions["word"])):
    emotiondic[emotions["word"][i]] = float(emotions["joy"][i])
#print(emotiondic)


def featurehash(trainall,scoretrain):
    for i in range(len(trainall)):
        temp = list(map(str,trainall[i]))
        sc = 0.0
        for j in temp:
            if j in hashfeat:
                sc += hashfeat[j]
        scoretrain[i].append(sc)

        
# Dictionary name define itself
emoticondic = {}
emoticon = open("emotican.txt","r")
for i in emoticon:
    tem = list(map(str,i.split()))
    emoticondic[tem[0]] = float(tem[1])
#print(emoticondic)

sent140dic = {}
sent140 = open("sent140.txt","r")
for i in sent140:
    tem = list(map(str,i.split()))
    #print(tem)
    sent140dic[tem[0]] = [float(tem[2]), float(tem[3])]
    
    
hashsentdic = {}
hashsent = open("hashsent.txt","r")
for i in hashsent:
    tem = list(map(str,i.split()))
    #print(tem)
    hashsentdic[tem[0]] = [float(tem[2]), float(tem[3])]
    
    
mpqadic = {}
mpqa = open("mpqa.txt","r")
for i in mpqa:
    tem = list(map(str,i.split()))
    mpqadic[tem[0]] = tem[0]
    
bingliudic = {}
bingliu = pandas.read_csv("bingliu.csv",sep = ",",error_bad_lines=False)
#print(bingliu)
for i in range(len(bingliu["word"])):
    bingliudic[bingliu["word"][i]] = bingliu["emotion"][i]
#print(bingliudic)
    

    


# preprocess function preprocessses the data which is used to make training score matrix and test score matrix
# 
# 
# 
# 

# In[5]:



def preprocess(train,trainall,result_train,scoretrain):
    print("Preprocessing start  ..........\n")
    print("making Score matrix")
    for i in train:
        scoretrain.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        temp = list(map(str,i.split()))
        #print(temp)
        
        result_train.append(float(temp.pop()))
        temp.pop(0)
        temp.pop()
        for j in temp:
            scoretrain[-1][0] += enlongated(j)
            scoretrain[-1][1] +=hashtag(j)
            scoretrain[-1][2] +=allcaps(j)
            scoretrain[-1][3] +=taggedp(j)
            scoretrain[-1][4] += negations(j)
            scoretrain[-1][5] += wordemo(j,wordemodic)
        scoretrain[-1][6] = hashemo(temp,hashdic)
        scoretrain[-1][7] = emotionsfeat(temp,emotiondic)
        
        scoretrain[-1].append(emoticonfeat(temp,emoticondic))
        aaa0,aaa1 = sent140feat(temp,sent140dic)
        scoretrain[-1].append(aaa0)
        scoretrain[-1].append(aaa1)
        aaa0,aaa1 = hashsentfeat(temp,hashsentdic)
        scoretrain[-1].append(aaa0)
        scoretrain[-1].append(aaa1)
        trainall.append(" ".join(temp))
        aaa0,aaa1 = bingliufeat(temp,bingliudic)
        scoretrain[-1].append(aaa0)
        scoretrain[-1].append(aaa1)
        aaa1,aaa2 = mpqafeat(temp,mpqadic)
        scoretrain[-1].append(aaa0)
        scoretrain[-1].append(aaa1)
        
        aaa0,aaa1 = puncfeat(i)
        scoretrain[-1].append(aaa0)
        scoretrain[-1].append(aaa1)
        
        scoretrain[-1].append(negatfeat(i))
        

        
        
        
    vader(scoretrain,trainall)
    


# For printing statistics for particular model including Pearson and spearman coff. 

# In[6]:



def statis(result_test,sss):
    mae = metrics.mean_absolute_error(result_test, sss)    
    mse = metrics.mean_squared_error(result_test, sss)     
    rmse = np.sqrt(mse) #mse**(0.5)      # 
    r2 = metrics.r2_score(result_test, sss)

    print("Results of sklearn.metrics:")
    print("MAE:",mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)
    print("pearson corr. , p valve =",stats.pearsonr(result_test,sss))
    print(stats.spearmanr(result_test,sss),"\n\n")


# In[ ]:





# In[7]:


hashfeat = {}    # DICTIONARY FOR hashtags predicting nature for sentence 

print("Training loading ..........\n")
train = open("joy_train.txt", "r")  # open file 
trainall = []  # contain training sentences
result_train = []   # result for training
scoretrain = []    # features for training data
preprocess (train,trainall,result_train,scoretrain)   # preprocessing

for i in range(len(trainall)):
    temp = list(map(str,trainall[i]))
    for j in temp:
        if j[0] == "#":
            if j in hashfeat:
                if result_train[i] > 0.55:
                    hashfeat[j] += 1
                else:
                    hashfeat[j] -= 1
            else:
                if result_train[i] > 0.55:
                    hashfeat[j] = 1
                else:
                    hashfeat[j] = -1

def featurehash(trainall,scoretrain):
    for i in range(len(trainall)):
        temp = list(map(str,trainall[i]))
        sc = 0.0
        for j in temp:
            if j in hashfeat:
                sc += hashfeat[j]
        scoretrain[i].append(sc)
featurehash(trainall,scoretrain)    # adding 13th feature

#for i in trainall:
print(len(scoretrain[0]))
#    print(i)
        
        
    
     


# In[8]:


print("Testing loading ..........\n")

test = open("joy_test.txt", "r")
testall = []  # testing sentences
result_test = []  # test results
scoretest  = []  # features for testing



preprocess(test,testall,result_test,scoretest)
featurehash(testall,scoretest)
    
     


# In[9]:


print("Converting to numpy.......\n")

score_trainn = np.array(scoretrain)
score_testn = np.array(scoretest)
result_trainn = np.array(result_train) 
result_testn = np.array(result_test) # making numpy list

#print(scoretrain)


# In[ ]:





# ### NGRAMS (Predicting model using only ngram features)

# In[10]:


print("Predict Model Uing only ngram (1,2) .......\n")


co_vect = CountVectorizer(ngram_range=(1,2))
final_tr  = co_vect.fit_transform(trainall)
final_te = co_vect.transform(testall)


# In[11]:


print("SVR Model\n")
modelprid(SVR(),final_tr,final_te,result_trainn,result_test)    # Using SVR


# In[12]:


print("MLP Model")
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,max_iter=5000) # arguments makes it very fast

modelprid(clf,final_tr,final_te,result_trainn,result_test) # using MLP


# In[13]:

print("Decision Tree Model")

modelprid(DecisionTreeRegressor(max_depth = 5),final_tr,final_te,result_trainn,result_test) # Using Decision tree


# In[14]:



print("\n\nPredicting Model using 25 features ")


# ### Predicting Models using our 25 features 

# In[15]:


print("SVR Model\n")

modelprid(SVR(),score_trainn,score_testn,result_trainn,result_test) # Using SVR


# In[16]:


print("MLP Model\n")

modelprid(clf,score_trainn,score_testn,result_trainn,result_test)   # Using MLP


# In[17]:


print("Decision Tree Model\n")

modelprid(DecisionTreeRegressor(max_depth = 5),score_trainn,score_testn,result_trainn,result_test)  # Using Decision Tree


# In[18]:


print("\n\nPredicting Model using N Gram + 25 features ")


# ### Predicting model by adding Ngrams and 25 features

# In[19]:



feature_matrix1 = final_tr.toarray()
feature_matrix2 = final_te.toarray()      # Convert sparse to numpy

print("Concating all features....")
score_trainnn = np.concatenate(( score_trainn, feature_matrix1), 1)  
score_testnn = np.concatenate((score_testn,feature_matrix2), 1)      # adds Ngram numpy and 13 features matrix


# In[20]:


print("SVR Model\n")

modelprid(SVR(),score_trainnn,score_testnn,result_trainn,result_test)  # Using SVR


# In[21]:


print("Decision Tree Model\n")

modelprid(DecisionTreeRegressor(max_depth = 5),score_trainnn,score_testnn,result_trainn,result_test)  # Using Decision Tree 


# In[22]:


clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,max_iter=2000) # arguments makes it very fast
print("MLP Model\n")

modelprid(clf,score_trainnn,score_testnn,result_trainn,result_test)  # using MLP
   


# https://cmdlinetips.com/2019/08/how-to-compute-pearson-and-spearman-correlation-in-python/

# Reason for having nan values in Pearson and Spearman coff.  
#   
#   https://datascience.stackexchange.com/questions/10262/why-is-the-correlation-coefficient-of-a-constant-function-with-function-input-is
# 

# In[ ]:



    


# In[ ]:





# In[ ]:




