import nltk, re, string, collections, glob
from nltk.util import ngrams
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("wordnet")

# defined path
path = "op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold1/*.txt"
# glob is used to go to every file in the directory
files = glob.glob(path)

text = ""
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
