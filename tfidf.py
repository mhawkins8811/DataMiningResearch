import nltk, re, string, collections, glob, math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob as tb


def normalize(tokenized):
    index = 0
    for token in tokenized:
        tokenized[index] = token.lower()
        index += 1
    return tokenized


def myStemmer(mysentencestokens):
    porter = nltk.PorterStemmer()
    index = 0
    # Stem words
    for token in mysentencestokens:
        mysentencestokens[index] = porter.stem(token)
        index += 1
    return mysentencestokens


def myLemma(mysentencestokens):
    lemma = nltk.stem.wordnet.WordNetLemmatizer()
    index = 0
    for token in mysentencestokens:
        mysentencestokens[index] = lemma.lemmatize(token)
        index += 1
    return mysentencestokens



punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"


paths = ["positive_polarity/truthful_from_TripAdvisor", "positive_polarity/deceptive_from_MTurk",
         "negative_polarity/truthful_from_Web", "negative_polarity/deceptive_from_MTurk"]
text = ""
bloblist = []
for folderName in paths:
    print("Working on "+folderName)
    for i in range(1, 6):
        # defined path
        # Positive and truthful
        path = "op_spam_v1.4/"+folderName+"/fold"+str(i)+"/*.txt"

        # glob is used to go to every file in the directory
        files = glob.glob(path)

        # put all the files on text
        for name in files:
            with open(name) as file:
                text = file.read()
                text = re.sub(punctuationNoPeriod, "", text)
                tokenized = text.split()

                tokenized = normalize(tokenized)

                minleng = 2
                mysentencestokens = [word for word in tokenized if(word not in stopwords.words("english")) and len(word) >= minleng]

                mysentencestokens = myStemmer(mysentencestokens)

                mysentencestokens = myLemma(mysentencestokens)

                review = " ".join(mysentencestokens)

                bloblist.append(review)

    print(folderName+" Done!")

vectorize = TfidfVectorizer()
matrix = vectorize.fit_transform(bloblist).todense()

matrix = pd.DataFrame(matrix, columns=vectorize.get_feature_names())

top_words = matrix.sum(axis=0).sort_values(ascending=False)

print(top_words)


