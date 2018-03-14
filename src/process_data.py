import csv
import gensim
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from collections import defaultdict

f = open('amazonReviewElectronicShortCSV.csv','r')

csvdata = csv.reader(f)

#Index listing
# 0 - reviewerID
# 1 - asin
# 2 - reviewerName
# 3 - helpful
# 4 - reviewText (Use this as corpus?)
# 5 - overall
# 6 - summary
# 7 - unixReviewTime
# 8 - reviewTime

#Generate corpus
documents = []
rowcount = 0
for row in csvdata:
    if rowcount > 0:
        documents.append(row[4])
    rowcount+=1
    
f.close()

#Remove stop words
stoplist = set(nltk.corpus.stopwords.words('english'))
stoplist.update(['-'])

texts = [[ word for word in document.lower().split() if word not in stoplist]
         for document in documents]

#pprint (texts[0])
#print '\n\n\n'

# Remove tokens with single occurrence
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text
          if frequency[token] > 1 and token.isalpha()]
         for text in texts]

#pprint texts[0]

#Store dictionary as binary/txt using gensim
dictionary = gensim.corpora.Dictionary(texts)
#dictionary.save('./dict/amazon_electronic_review.dict')
dictionary.save_as_text('./dict/amazon_slectronic_review_text.txt')

#pprint(dictionary.token2id)
#print(dictionary.token2id)

doc_bow = [dictionary.doc2bow(text) for text in texts]
print(texts[0])
print(doc_bow[0])

#
# TF-IDF Vectorizing using scikitlearn
#

#vectorizer = TfidfVectorizer(stop_words=stoplist, use_idf=True)
#V = vectorizer.fit_transform(documents)
#pprint(V)
#print V

print '\n\n==END==\n\n'
