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
from collections import Counter

# Training dataset
partiallyTaggedDataset = open("dataset/CRFdataset_trigrams.txt").read().split('\n')
# Dev test set
devTestDataset = open("dataset/Dev_testset.txt").read().split('\n')

# Loading spacy en library
nlp = spacy.load('en')


words = []
wordsWithFeatures = []
rawWords = []

# Splitting the taggedDataset to Array of sentence and words
def dataset2WordArray(dataset):
    wordsWithFeatures = []
    for line in dataset:
        splittedLines = line.split(',')
        sentWithSplittedWords = []
        for taggedWords in splittedLines:
            splittedWords = taggedWords.split(' ')
            sentWithSplittedWords.append(splittedWords)
        wordsWithFeatures.append(sentWithSplittedWords)
    return wordsWithFeatures

# POS tag (Spacy) and First word feature is appended to each word of the sentence array
def addFeatures2Words(wordsWithFeatures):
    for i in xrange(0, len(wordsWithFeatures)):
        sentence = " ".join([t for t,r in wordsWithFeatures[i]]) # converting the tagged dataset to a sentence for retrieving pos tags
        doc = nlp(unicode(sentence))
        for j in xrange(0,len(doc)):
            wordsWithFeatures[i][j].insert(1,doc[j].tag_)
            if j == 0:
                wordsWithFeatures[i][j].insert(1, '1')
            else:
                wordsWithFeatures[i][j].insert(1, '0')
    return wordsWithFeatures

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
        'postag=' + postag
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
            '-1:postag=' + postag1
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
            '+1:postag=' + postag1
            ])
    else:
        features.append('EOS')

    return features

def sent2features(sent):
    return [wordsToFeatures(sent, i) for i in range(len(sent))]

#  Extracting lables from sentence array
def sent2labels(sent):
    return [label for token, isFirstWord, postag, label in sent]

#  Extracting words alone from sentence array
def sent2tokens(sent):
    return [token for token, isFirstWord, postag, label in sent]

# print words[1]
# print wordsInFeatures[0]
wordsWithFeatures = dataset2WordArray(partiallyTaggedDataset)
wordsWithFeatures = addFeatures2Words(wordsWithFeatures)
devTestWithFeatures = dataset2WordArray(devTestDataset)
devTestWithFeatures = addFeatures2Words(devTestWithFeatures)

print len(wordsWithFeatures)
random.seed(3)
random.shuffle(wordsWithFeatures)
train = wordsWithFeatures

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


X_test = [sent2features(sent) for sent in devTestWithFeatures]
y_test = [sent2labels(sent) for sent in devTestWithFeatures]

crfsuiteTrainer = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1,
                                       c2=0.1,
                                       max_iterations=100,
                                       all_possible_transitions=True)
crfsuiteTrainer.fit(X_train,y_train)

y_pred = crfsuiteTrainer.predict(X_test)
labels = list(crfsuiteTrainer.classes_)
print metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(crfsuiteTrainer.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crfsuiteTrainer.state_features_).most_common()[-30:])



sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

for i in xrange(0,len(devTestDataset)):
    print " ".join([t for t,a,b,r in devTestWithFeatures[i]])
    print y_test[i]
    print y_pred[i]

# print test[4],y_train[4]
# print y_pred[4]
# print X_train[4]

while(True):
    query = raw_input(">>")
    wordsWithFeatures = []
    rawDataset =[]
    rawDataset.append(query)
    splittedWords = query.split(' ')
    wordsWithFeatures.append([list(z) for z in zip(splittedWords, [''] * len(splittedWords))])
    addFeatures2Words(wordsWithFeatures)
    testQuery =  [sent2features(sent) for sent in wordsWithFeatures]
    print crfsuiteTrainer.predict(testQuery)



