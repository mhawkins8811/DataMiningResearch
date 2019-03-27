import nltk, re, string, collections, glob
from nltk.util import ngrams
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download("stopwords")
#nltk.download("wordnet")
import csv


#lines to read and bring inthe csv file
#with open('deceptive-opinion.csv') as csvfile:
#    readCSV = csv.reader(csvfile, delimiter=',')
#    for row in readCSV:
#        text = csvfile.read()

text = ""

for i in range(1,6):
    # defined path
    # Positive and truthful
    # path = "op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold"+str(i)+"/*.txt"

    # Positive and deceptive
    # path = "op_spam_v1.4/positive_polarity/deceptive_from_MTurk/fold" + str(i) + "/*.txt"

    # Negative and truthful
    # path = "op_spam_v1.4/negative_polarity/truthful_from_Web/fold" + str(i) + "/*.txt"

    # Negative and deceptive
    path = "op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold" + str(i) + "/*.txt"

    # glob is used to go to every file in the directory
    files = glob.glob(path)

    # put all the files on text
    for name in files:
        with open(name) as file:
            text += file.read()

# Remove punctuation
punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"
text = re.sub(punctuationNoPeriod, "", text)

# Split to tokenize all the words on text
tokenized = text.split()

index = 0
for token in tokenized:
    tokenized[index] = token.lower()
    index += 1


# To remove small words
minleng = 2
# Remove stop words
mysentencestokens = [word for word in tokenized if(word not in stopwords.words("english")) and len(word) >= minleng]

# Initialize stemmer
porter = nltk.PorterStemmer()
index = 0
# Stem words
for token in mysentencestokens:
    mysentencestokens[index] = porter.stem(token)
    index += 1

# Lemmatizer to guarantee that they are words
lemma = nltk.stem.wordnet.WordNetLemmatizer()

index = 0
for token in mysentencestokens:
    mysentencestokens[index] = lemma.lemmatize(token)
    index += 1

# Create n grams
unigram_val = nltk.ngrams(mysentencestokens, 1)

bigram_val = nltk.bigrams(mysentencestokens)

trigram_val = nltk.trigrams(mysentencestokens)

# Get their frequencies
unigram_val_freq = collections.Counter(unigram_val)

bigram_val_freq = collections.Counter(bigram_val)

trigram_val_freq = collections.Counter(trigram_val)

# Print them
print("Unigram:")
print(unigram_val_freq.most_common(10))
print("\n")
print("Bigram:")
print(bigram_val_freq.most_common(10))
print("\n")
print("Trigram:")
print(trigram_val_freq.most_common(10))
print("\n")

with open("save/bigramFalseNegative.csv", 'w') as file:
    for value in bigram_val_freq.items():
        file.write(str(value)+"\n")
    file.close()

with open("save/unigramFalseNegative.csv", 'w') as file:
    for value in unigram_val_freq.items():
        file.write(str(value)+"\n")
    file.close()

with open("save/trigramFalseNegative.csv", 'w') as file:
    for value in trigram_val_freq.items():
        file.write(str(value)+"\n")
    file.close()
