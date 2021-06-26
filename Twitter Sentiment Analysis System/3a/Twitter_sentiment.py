#!/usr/bin/env python
# coding: utf-8

# Follow Instructions:
# + Features Implemented can be seen in scoremat function.
# + We implement all the features + Addtional features and we can include or exclude them by directly changing scoremat function.
# + Program is optimized

# Import Files

# In[1]:


import pandas
from time import time
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from os import path
import re
import nltk
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
from collections import Counter


# Functions for features

# In[2]:


def enlongated(st):   # elongated Feature
    temp = re.sub(r'(.)\1+', r'\1\1', st)
    if len(temp) == len(st):
        return 0
    else:
        return 1

def hashtag(st):     # Count Hashtag
    if st[0] == "#":
        return 1
    return 0 

def allcaps(st):     # count all capital words
    if st.isupper():
        return 1
    return 0

def negations(st):    # to count negations(BONUS)
    if st.lower() in ["not","nor","neither","no","never","nope"]:
        return 1
    return 0

def emoticonfeat(st,dic):  # Emoticon feature
    aa = 0
    for i in st:
        if i in dic:
            aa += dic[i]
    return aa


def sent140feat(st,dic):    # lexicon features
    posi = []
    neg = []
    for i in st:
        if i in dic:
            posi.append(dic[i][0])
            neg.append(dic[i][1])
    if posi == []:
        return 0,0
    return sum(posi)/len(posi) , sum(neg)/len(neg)

def hashsentfeat(st,dic):    # lexicon feature
    posi = []
    neg = []
    for i in st:
        if i in dic:
            posi.append(dic[i][0])
            neg.append(dic[i][1])
    if posi == []:
        return 0,0
    return sum(posi)/len(posi) , sum(neg)/len(neg)

def wordemofeat(st,dic):         # Counts Emotion word count:
    aa = 0
    for i in st:
        if i.lower() in dic:
            aa += dic[i.lower()]
    return aa


def mpqafeat(st,dic):
    posi = 0
    neg = 0
    for i in st:
        if i.lower() in dic:
            if dic[i.lower()] == "positive":
                posi += 1
            elif dic[i.lower()] == "negative":
                neg += 1
    return posi, neg

def bingliufeat(st,dic):
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
    
def puncfeat(i):    # Punctuation feature
    if i[-1] == "!" or i[-1]=="?":
        return len(re.findall("[!?]{2,}",i)) , 1
    else:
        return len(re.findall("[!?]{2,}",i)) , 0
    
    
def negatfeat(i):  # Negation feature
    return len(re.findall(r"(?i)(\b(no|none|not|nothing|neither|never|can't|isn't|doesn't|wouldn't|shouldn't)\b).*(,|\.|:|;|!|\?)",i))    

    


def tagger(tag,count):   
    if tag in count:
        return count[tag]
    return 0
    

def POS(feature,i):  # POS feature
    tags = nltk.pos_tag(i.split())
    counts = Counter( tag for word,  tag in tags)
    feature.append(tagger("NN",counts))
    feature.append(tagger("DT",counts))
    feature.append(tagger("VBZ",counts))
    feature.append(tagger("JJ",counts))
    feature.append(tagger("RB",counts))
    feature.append(tagger("JJ",counts))
    feature.append(tagger("CC",counts))
    feature.append(tagger("PRP",counts))
    feature.append(tagger("TO",counts))
    feature.append(tagger("IN",counts))
    feature.append(tagger("VB",counts))
    feature.append(tagger("NNP",counts))
    



    #print(accu)
    #print(len(accu))
    
    


# Functions For stats 

# In[3]:


def statsnaive(result_test,pred_test):
    accu = []
    for i in range(len(result_test)):
        if result_test[i] == pred_test[i]:
            accu.append(1)
        else:
            accu.append(0)
    print("accuracy = ", sum(accu) * 100 / len(accu))

def statis(test,pred):   # FInal function for statistics
    
    statsnaive(test,pred)
    print("0 represents negative sentiment and 1 represents positive sentiment")
    confu_matrix = [[0,0],[0,0]]
    for i in range(len(test)):
        if test[i] == 4 and pred[i] == 4:
            confu_matrix[1][1] += 1
        if test[i] == 4 and pred[i] == 0:
            confu_matrix[1][0] += 1
        if test[i] == 0 and pred[i] == 4:
            confu_matrix[0][1] += 1
        if test[i] == 0 and pred[i] == 0:
            confu_matrix[0][0] += 1
    print("Confusion matrix (row represents original result and col represents predicted results :-")
    print("   0     1")
    print(0,*confu_matrix[0])
    print(1,*confu_matrix[1])
    
    tp0 = confu_matrix[0][0]
    fp0 = confu_matrix[1][0]
    fn0 = confu_matrix[0][1]
    
    tp1 = confu_matrix[1][1]
    fp1 = confu_matrix[0][1]
    fn1 = confu_matrix[1][0]
    
    print("\nFor negative sentiment:")
    prec0 = tp0 / (tp0 + fp0)
    recall0 = tp0 / (tp0 + fn0)
    f1s0 = 2 * (prec0 * recall0) / (prec0 + recall0)
    print("precsion =",prec0)
    print("recall = ",recall0)
    print("f1s = ",f1s0)
    
    
    print("\nFor positive sentiment:")
    prec1 = tp1 / (tp1 + fp1)
    recall1 = tp1 / (tp1 + fn1)
    f1s1 = 2 * (prec1 * recall1) / (prec1 + recall1) 
    print("precsion =",prec1)
    print("recall = ",recall1)
    print("f1s = ",f1s1)
    
    print("Aver. Precision = ",(prec0 +prec1)/2)
    print("Aver. recall = ",(recall0 + recall1)/2)
    print("Aver. f1s = ", (f1s0 + f1s1)/2)
    
    
    
    
    
    
        


# Dictionaries for Features containing respective values
# Used in scoremat function

# In[4]:


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
    
    
wordem = open("wordemotion.txt","r")
wordemodic = {}      # count words for particular emotion (anger or joy)
for i in wordem:
    tem = list(map(str,i.split()))
    if tem[1] == "joy" and tem[2] == "1":
        wordemodic[tem[0]] = 1
        
        
    
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
    


# Preprocessing to made feature matrix

# In[5]:


def scoremat(scoretrain,trainfile,probyes):
    senten = trainfile["5"].tolist()
    result_train = trainfile["0"].tolist()
    for i in senten:
        temp = list(map(str,i.split()))
        nv_posi,nv_neg = naiveb(temp,probyes)
        scoretrain.append([nv_posi,nv_neg,0,0,0,0])
        st = list(map(str,i[1].split()))
        for j in st:
            scoretrain[-1][2] += enlongated(j)
            scoretrain[-1][3] += hashtag(j)
            scoretrain[-1][4] += allcaps(j)
            scoretrain[-1][5] += negations(j)
        scoretrain[-1].append(emoticonfeat(temp,emoticondic))
 
        #aaa0,aaa1 = sent140feat(temp,sent140dic)   # significant change in accuracy and time
        #scoretrain[-1].append(aaa0)
        #scoretrain[-1].append(aaa1)
        #aaa0,aaa1 = hashsentfeat(temp,hashsentdic)  # significant change in accuracy and time
        #scoretrain[-1].append(aaa0)
        #scoretrain[-1].append(aaa1)
        
        
        #trainall.append(" ".join(temp))
        
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
        POS(scoretrain[-1],i.lower())
        #print(len(scoretrain))
        
        #scoretrain[-1].append(wordemofeat(temp,wordemodic)) # reduce 2 -3 percent accuracy
        
        
    return scoretrain,result_train


# Naive bayes from scratch 

# In[6]:



def naiveb(st,probyes):     # naive bayes
    scp = 1 
    scn = 1
    probno = 1-probyes
    #print(st)
    for i in st:
        if i in dictword:
            scp*= dictword[i][1]/probyes
            scn *= dictword[i][0]/probno
    return scp / (scp + scn) , scn/(scp+scn)


def naivebres(scoretest):
    pred_test = []
    for i in scoretest:
        if i[0] > i[1]:
            pred_test.append(4)
        else:
            pred_test.append(0)
        #print(i[0],i[1])
    return pred_test

            
def statsnaive(result_test,pred_test):
    accu = []
    for i in range(len(result_test)):
        if result_test[i] == pred_test[i]:
            accu.append(1)
        else:
            accu.append(0)
    print("accuracy = ", sum(accu) * 100 / len(accu))


# Main code Starts from here

# In[7]:


smoothing = 1    # Take smoothing for dictionary as 1
start_time = time()   

scoretrain = []  # scoring matrix (features)
result_train = [] 

trainfile = pandas.read_csv('strainfull.csv', sep=',')
result_train = trainfile["0"].tolist()

probyes = (sum(result_train) / 4 ) / len(result_train)


print("Making bag of words.....")
if path.isfile('ser_dict.txt'):
    dictword = pickle.load(open("ser_dict.txt" ,"rb"))
else:
    dictword = {}      # Bag of Words 
    for ttt in range(len(trainfile["5"])):
        if ttt%100000 == 0:
            print(ttt)
        i = [trainfile["0"][ttt],trainfile["5"][ttt]]
        st = list(map(str,i[1].split()))
        #print(st)
        for j in st:
            if j in dictword:
                if i[0] == 4:
                    dictword[j][1] += 1
                else:
                    dictword[j][0] += 1
            else:
                dictword[j] = [smoothing,smoothing]
                if i[0] == 4:
                    dictword[j][1] += 1
                else:
                    dictword[j][0] += 1
    print(len(dictword))
    with open('ser_dict.txt', 'wb') as fh:
        pickle.dump(dictword,fh)
        


# In[8]:


co_vect = CountVectorizer(ngram_range=(1,2))  # preparing ngram count_vectorizer dictionary
co_vect.fit(trainfile["5"][:300000]) # cacn be increased or decreased


# In[9]:


trainfile = pandas.read_csv('strain.csv', sep=',')   

print("Train loading........")
scoretrain , result_train =  scoremat(scoretrain,trainfile,probyes)
trainall = trainfile["5"].tolist()# store training sentences


# In[10]:



scoretest = []   # scoring matrix
result_test = []

print("testing loading")
testfile = pandas.read_csv('stest.csv', sep=',')
testfile = testfile[:1500]
testall = testfile["5"].tolist()

print("Prepare Scoring matrix")
scoretest ,result_test = scoremat(scoretest,testfile,probyes)


# In[11]:


print("Results for Naive Bayes..... \n")
pred_test = naivebres(scoretest)   # Naive Bayes 
statis(result_test,pred_test)      # Stats for Naive Bayes


# Making Numpy arrays

# In[12]:


scoretrain = np.array(scoretrain)
scoretest = np.array(scoretest)


# Concatinate feature matrix with Ngrams

# In[13]:


final_tr  = co_vect.fit_transform(trainall)
feature_matrix1 = final_tr.toarray()
score_train = np.concatenate((scoretrain, feature_matrix1), 1)

model1 = SVC()
model2 = DecisionTreeClassifier(max_depth = 5)
model3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1,max_iter=5000)


# Making SVC Model

# In[18]:


print("Model 1..........")
if path.isfile('model1.txt'):
    model1 = pickle.load(open("model1.txt" ,"rb"))
else:
    with open('model1.txt', 'wb') as fh:
        model1.fit(score_train, result_train)
        pickle.dump(model1,fh)
        
    
    


# Making Decision Tree model

# In[19]:



print("Model 2..........")
if path.isfile('model2.txt'):
    model2 = pickle.load(open("model2.txt" ,"rb"))
else:
    with open('model2.txt', 'wb') as fh:
        model2.fit(score_train, result_train)
        pickle.dump(model2,fh)


# Making MLP model

# In[20]:


print("Model 3..........")
if path.isfile('model3.txt'):
    model3 = pickle.load(open("model3.txt" ,"rb"))
else:
    with open('model3.txt', 'wb') as fh:
        model3.fit(score_train, result_train)
        pickle.dump(model3,fh)


# Here we are predicting our model by spliting test data into 6000 size's slits
# It reduces space complexity 
# Decreases time
# No change in accuracy

# In[ ]:





# In[ ]:





# In[21]:



pred = []
start_time = time()
gapp = 1500  # Breaking test splits 
svc = []  # svc results
dt = []   # dt results
mlp = []   # mlp results

for i in range(0,len(scoretest),gapp):
    print("Testing on ",i ,"to", i + gapp)
    result_testl = result_test[i:i+gapp]
    final_te = co_vect.transform(testall[i:i+gapp])
    feature_matrix = final_te.toarray() 
    score_testnn = np.concatenate((scoretest[i:i+gapp],feature_matrix),1)  # Concating All features
    svc += list(model1.predict(score_testnn))             # SVC Results
    dt +=  list(model2.predict(score_testnn))            # Decision Tree Results
    mlp += list(model3.predict(score_testnn))              # For MLP  
    #print(len(dt))
    
    


    
    
    
    
    


# In[22]:


print("\n\n\nStats For SVC Model")
statis(result_test,svc)


# In[23]:



print("\n\n\nStats For Decision Tree Model")
statis(result_test,dt)


# In[24]:


print("\n\n\nStats For MLP Model")
statis(result_test,mlp)


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




