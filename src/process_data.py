import csv
import gensim
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


#Store dictionary as binary/txt using gensim
dictionary = gensim.corpora.Dictionary(texts)
#dictionary.save('./dict/amazon_electronic_review.dict')
dictionary.save_as_text('./dict/amazon_slectronic_review_text.txt')


#
# TF-IDF Vectorizing using scikitlearn
#

tfidf_vectorizer = TfidfVectorizer(stop_words=stoplist, use_idf=True)
V = tfidf_vectorizer.fit_transform(documents)

#
# Mapping feature score to actual words in doc
#
#document_number=0
#feature_names = tfidf_vectorizer.get_feature_names()
#feature_index = V[document_number,:].nonzero()[1]
#tfidf_scores = zip(feature_index, [V[document_number, x] for x in feature_index])
#for word, score in [(feature_names[index], score) for (index, score) in tfidf_scores]:
#  print str(word) + ' => ' + str(score)

#
# Calculating the pairwise cosine similarity for each document in corpus
#

cs_results = cosine_similarity(V[0:3], V)
doc_num = 1
for i_result in cs_results:
    print "Document#: ",doc_num,'\n'
    print i_result,'\n\n'
    doc_num += 1


print '\n\n==END==\n\n'
