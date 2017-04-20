from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import spacy
import random

partiallyTaggedDataset = open("dataset/CRFdataset.txt").read().split('\n')
rawDataset = open("dataset/CRFdatasetWithoutTags.txt").read().split('\n')
devTest = open("dataset/Dev_testset.txt").read().split('\n')

nlp = spacy.load('en')


words = []
wordsInFeatures = []
rawWords = []

# Splitting the taggedDataset to Array of sentence and words
for line in partiallyTaggedDataset:
    splittedLines = line.split(',')
    words.append(splittedLines)

# Partiallytaggeddataset is split into words with features
for sent in words:
    sentWithSplittedWords = []
    for taggedWords in sent:
        splittedWords = taggedWords.split(' ')
        sentWithSplittedWords.append(splittedWords)
    wordsInFeatures.append(sentWithSplittedWords)

# POS tag (Spacy) and First word feature is appended to each word of the sentence array
def addFeatures2Words(wordsInFeatures):
    for i in xrange(0,len(wordsInFeatures)):
        doc = nlp(unicode(" ".join([t for t,r in wordsInFeatures[i]])))  # converting the tagged dataset to a sentence for retrieving pos tags
        for j in xrange(0,len(doc)):
            wordsInFeatures[i][j].insert(1,doc[j].tag_)
            if j == 0:
                wordsInFeatures[i][j].insert(1,'1')
            else:
                wordsInFeatures[i][j].insert(1,'0')


addFeatures2Words(wordsInFeatures)

def wordsToFeatures(sent,i):
    word = sent[i][0]
    isFirstWord = sent[i][1]
    postag = sent[i][2]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        # 'word.isFirstWord='+ isFirstWord,
        'postag=' + postag[:2]
        ]
    if i > 0:
        word1 = sent[i-1][0]
        isFirstWord1 = sent[i-1][1]
        postag1 = sent[i-1][2]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            # '-1:word.isFirstWord='+ isFirstWord1,
            '-1:postag=' + postag1[:2]
            ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        isFirstWord1 = sent[i+1][1]
        postag1 = sent[i+1][2]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            # '+1:word.isFirstWord='+ isFirstWord1,
            '+1:postag=' + postag1[:2]
            ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
    return [wordsToFeatures(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, isFirstWord, postag, label in sent]

def sent2tokens(sent):
    return [token for token, isFirstWord, postag, label in sent]

# print words[1]
# print wordsInFeatures[0]

print len(wordsInFeatures)
random.seed(3)
random.shuffle(wordsInFeatures)
train = wordsInFeatures
# test = wordsInFeatures[:2988]

# for i in xrange(0,len(train)):
#     for j in xrange(0,len(train[i])):
#         w = train[i][j]
#         if w[3] == 'SRCH' and (w[2] == 'IN' or w[2] == 'JJ'):
#             train[i][j][2] = 'NN'
#
# for i in xrange(0,len(train)):
#     for w in train[i]:
#         if w[3]=='SRCH':
#             print w[0],w[2]

X_train =  [sent2features(sent) for sent in train]
y_train = [sent2labels(sent) for sent in train]



# X_test = [sent2features(sent) for sent in test]
# y_test = [sent2labels(sent) for sent in test]

crfsuiteTrainer = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1,
                                       c2=0.1,
                                       max_iterations=100,
                                       all_possible_transitions=True)
crfsuiteTrainer.fit(X_train,y_train)

# y_pred = crfsuiteTrainer.predict(X_test)

labels = list(crfsuiteTrainer.classes_)
# print metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)
# print test[4],y_train[4]
# print y_pred[4]
# print X_train[4]

while(True):
    query = raw_input(">>")
    wordsInFeatures = []
    rawDataset =[]
    rawDataset.append(query)
    splittedWords = query.split(' ')
    wordsInFeatures.append([list(z) for z in zip(splittedWords,['']*len(splittedWords))])
    addFeatures2Words(wordsInFeatures)
    testQuery =  [sent2features(sent) for sent in wordsInFeatures]
    print crfsuiteTrainer.predict(testQuery)



