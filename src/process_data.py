import pandas as pd
import nltk

#Test comment

from html2text import unescape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

kmeanColors = ['red', 'green', 'blue', 'purple', 'orange']
#kmeanColors = ['blue', 'blue', 'blue', 'blue', 'blue']
#kmeanColors = ['red', 'green', 'blue']

# Load stop words to reduce noise
stoplist = set(nltk.corpus.stopwords.words('english'))
stoplist.update(['-', 'yet', 'yea', 'zs15'])

############################
# Helper fuctions + Values #
############################

#Wait for plot to close before continuing with the script
#Set to true if you want to see hover effects and interact with plot
#If false, will just print out all plots at the end

blockPlotting = False

# Hover over annotation/labeling.
def update_annot(ind):
    
    pos = sc.get_offsets()[ind['ind'][0]]
    annot.xy = pos
    text = 'Review#: {}'.format(ind['ind'][0])
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(1)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

def print_top_features(cluster_number, dataframe):
    
    cluster=dataframe.loc[dataframe['cluster']==cluster_number]
    cluster_tfidf = TfidfVectorizer(stop_words=stoplist, use_idf=True)

    V_cluster = cluster_tfidf.fit_transform(cluster.reviewText)
    
    indices = np.argsort(cluster_tfidf.idf_)[::-1]
    features = cluster_tfidf.get_feature_names()
    top_n = 5
    top_features = [features[i] for i in indices[:top_n]]
    print('{} - {}'.format(kmeanColors[cluster_number],top_features))


#
# START PROCESSING
#

# Load the review/text corpus
df=pd.read_csv('../datasets/testdata.csv')
df=df[df.reviewText.str.len() > 5]
df.reviewText = df.reviewText.apply(unescape, unicode_snob=True)


# Create tf-idf vectorizer, using SKLearn
tfidf_vectorizer = TfidfVectorizer(stop_words=stoplist, use_idf=True)
V = tfidf_vectorizer.fit_transform(df.reviewText)


#
# The below is primairly for calculating the cosine similarity and plotting the top features (words)
# with the highest tfidf scores for each documents.
#
# Calculating the pairwise cosine similarity for each document in corpus
# Uncomment below if you wish to view cosine similarities, and top occurring words.
# Warning* Expensive operation

#
# tf-idf score output
#
# Mapping feature names (words) to its tfidt scores for each documents (reviews)
# And print results.

#This is the document number (0... to max length of dataset processed)
#review_number=0

#feature_names = tfidf_vectorizer.get_feature_names()
#feature_index = V[review_number].nonzero()[1]
#tfidf_scores = zip(feature_index, [V[review_number, x] for x in feature_index])
#for word, score in [(feature_names[index], score) for (index, score) in tfidf_scores]:
#  print (str(word) + ' => ' + str(score))


#
# cosine similarity output
#

cs_results = cosine_similarity(V[0:], V)
doc_num = 1
for i_result in cs_results:
    # Limiter
    if doc_num > 3:
        break;
    print ("Document#: ",doc_num)
    print (i_result,'\n\n')
    doc_num += 1

# TSNE - Reducing dimension size for above tf-idf scores to 2d/3d to be able to plot

#Embedd to dimension (value = {2 or 3}
embedded_dim = 2

#Run TSNE *THIS IS VERY EXPENSIVE OPERATION*
V_tsne = TSNE(n_components=embedded_dim, learning_rate=100).fit_transform(V.toarray())

#
# K-Means Clustering
#

#Use V if you want to run k-mean on the full unreduced vector.

km = KMeans(n_clusters=5, init='k-means++', n_init=50, max_iter=1000)
km.fit(V)
kmeanV = km.predict(V)



# Tag each document with it's respective cluster in the dataframe

df['cluster'] = kmeanV


# Print it's top term
print_top_features(0, df)
print_top_features(1, df)
print_top_features(2, df)
print_top_features(3, df)
print_top_features(4, df)

# Plot data in 2-D using matplotlib
# Grab all x and y coordiantes
x = V_tsne[:,0]
y = V_tsne[:,1]

points = V_tsne[:,0:0]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

sc = ax.scatter(x,y,c=kmeanV,cmap=colors.ListedColormap(kmeanColors))


fig.canvas.mpl_connect("motion_notify_event", hover)


plt.show(block=blockPlotting)

##################################################
# 3d scatter plotting section (WORK IN PROGRESS) #
##################################################

#
# K-Means Clustering
#

#Embedd to dimension (value = {2 or 3}
embedded_dim = 3

#Run TSNE
V_tsne = TSNE(n_components=embedded_dim, learning_rate=100).fit_transform(V.toarray())

fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection='3d')

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# TODO: Fix hover over annotation....
#       Or just figure out how to identify each point easily.
# fig3d.canvas.mpl_connect("motion_notify_event", hover)

xs = V_tsne[:,0]
ys = V_tsne[:,1]
zs = V_tsne[:,2]

ax.scatter(xs,ys,zs, c=kmeanV, cmap=colors.ListedColormap(kmeanColors))


plt.show(block=blockPlotting)

print ('\n\n==END==\n\n')
