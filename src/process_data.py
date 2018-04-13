import csv
import gensim
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import numpy as np

from pprint import pprint

f = open('amazonReviewElectronicShortCSV.csv','r')

#Full text
#f = open('../datasets/CSV_AMAZON_REVIEW_ELECTRONIC_FULL.csv', 'r')

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
#Run the nltk download below if stopword corpus is not yet downloaded.
#nltk.download()
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

#Shape of V
print "Shape of V:", V.shape

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
#cs_results = cosine_similarity(V[0:], V)
#doc_num = 1
#for i_result in cs_results:
#    #if doc_num % 1000 == 0:
#    print "Document#: ",doc_num,'\n'
#    print i_result,'\n\n'
#    doc_num += 1

#
# T-SNE Plotting
#
#vectorized_plot = []
#for x in range (0,99)
#print V[1, :].nonzero()[1]


V_tsne = TSNE(learning_rate=100).fit_transform(V.toarray())

print "Shape of V_tsne: ", V_tsne.shape

x = V_tsne[:,0]
y = V_tsne[:,1]

points = V_tsne[:,0:0]
color = np.sqrt((points**2).sum(axis=1))/np.sqrt(2.0)
rgb = plt.get_cmap('jet')(color)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y,color=rgb)
plt.show()

#
# K-Means Clustering
#

km = KMeans(n_clusters=7, init='k-means++', n_init=10, max_iter=300)
km.fit(V)

print km.predict(V)

print '\n\n==END==\n\n'
