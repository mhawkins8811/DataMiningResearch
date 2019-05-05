import nltk, re, string, collections, glob
from nltk.util import ngrams
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download("stopwords")#needed when ran for the first time then comment
#nltk.download("wordnet")#needed when ran for the first time then comment
import csv

textA = ""
textB = ""
textC = ""
textD = ""

#lines to read and bring inthe csv file
#with open('deceptive-opinion.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        text = csvfile.read()




for i in range(1,6):
    # defined path
    # Positive and truthful
    pathA = "op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold"+str(i)+"/*.txt"

    # Positive and deceptive
    pathB = "op_spam_v1.4/positive_polarity/deceptive_from_MTurk/fold" + str(i) + "/*.txt"

    # Negative and truthful
    pathC = "op_spam_v1.4/negative_polarity/truthful_from_Web/fold" + str(i) + "/*.txt"
    # Negative and deceptive
    pathD = "op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold" + str(i) + "/*.txt"


# glob is used to go to every file in the directory
    filesA = glob.glob(pathA)
    filesB = glob.glob(pathB)
    filesC = glob.glob(pathC)
    filesD = glob.glob(pathD)

    # put all the files on text
    for name in filesA:
        with open(name) as file:
            textA += file.read()
    for name in filesB:
        with open(name) as file:
            textB += file.read()
    for name in filesC:
        with open(name) as file:
            textC += file.read()
    for name in filesD:
        with open(name) as file:
            textD += file.read()





# Remove punctuation
punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
text = re.sub(punctuationNoPeriod, "", textA)

punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
text = re.sub(punctuationNoPeriod, "", textB)

punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
text = re.sub(punctuationNoPeriod, "", textC)

punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
text = re.sub(punctuationNoPeriod, "", textD)

# Split to tokenize all the words on text
tokenizedA = textA.split()
tokenizedB = textB.split()
tokenizedC = textC.split()
tokenizedD = textD.split()

indexA = 0
for token in tokenizedA:
    tokenizedA[indexA] = token.lower()
    indexA += 1

indexB = 0
for token in tokenizedB:
    tokenizedB[indexB] = token.lower()
    indexB += 1

indexC = 0
for token in tokenizedC:
    tokenizedC[indexC] = token.lower()
    indexC += 1

indexD = 0
for token in tokenizedD:
    tokenizedD[indexD] = token.lower()
    indexD += 1


# To remove small words
minleng = 2
# Remove stop words
mysentencestokensA = [word for word in tokenizedA if(word not in stopwords.words("english")) and len(word) >= minleng]
mysentencestokensB = [word for word in tokenizedB if(word not in stopwords.words("english")) and len(word) >= minleng]
mysentencestokensC = [word for word in tokenizedC if(word not in stopwords.words("english")) and len(word) >= minleng]
mysentencestokensD = [word for word in tokenizedD if(word not in stopwords.words("english")) and len(word) >= minleng]
# Initialize stemmer
porter = nltk.PorterStemmer()
# Stem words
indexA = 0
for token in mysentencestokensA:
    mysentencestokensA[indexA] = porter.stem(token)
    indexA += 1
indexB = 0
for token in mysentencestokensB:
    mysentencestokensB[indexB] = porter.stem(token)
    indexB += 1
indexC = 0
for token in mysentencestokensC:
    mysentencestokensC[indexC] = porter.stem(token)
    indexC += 1
indexD = 0
for token in mysentencestokensD:
    mysentencestokensD[indexD] = porter.stem(token)
    indexD += 1



# Lemmatizer to guarantee that they are words
lemma = nltk.stem.wordnet.WordNetLemmatizer()

indexA = 0
for token in mysentencestokensA:
    mysentencestokensA[indexA] = lemma.lemmatize(token)
    indexA += 1
indexB = 0
for token in mysentencestokensB:
    mysentencestokensB[indexB] = lemma.lemmatize(token)
    indexB += 1
indexC = 0
for token in mysentencestokensC:
    mysentencestokensC[indexC] = lemma.lemmatize(token)
    indexC += 1
indexD = 0
for token in mysentencestokensD:
    mysentencestokensD[indexD] = lemma.lemmatize(token)
    indexD += 1
# Create n grams
unigram_valA = nltk.ngrams(mysentencestokensA, 1)
unigram_valB = nltk.ngrams(mysentencestokensB, 1)
unigram_valC = nltk.ngrams(mysentencestokensC, 1)
unigram_valD = nltk.ngrams(mysentencestokensD, 1)


# Get their frequencies
unigram_val_freqA = collections.Counter(unigram_valA)
unigram_val_freqB = collections.Counter(unigram_valB)
unigram_val_freqC = collections.Counter(unigram_valC)
unigram_val_freqD = collections.Counter(unigram_valD)
# bigram_val_freq = collections.Counter(bigram_val)
#
# trigram_val_freq = collections.Counter(trigram_val)

# Print them
print("Positive and truthful:")
print(unigram_val_freqA.most_common(10))
print("Positive and deceptive:")
print(unigram_val_freqB.most_common(10))
print("Negative and truthful:")
print(unigram_val_freqC.most_common(10))
print("Negative and deceptive:")
print(unigram_val_freqD.most_common(10))
