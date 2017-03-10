import codecs
import string
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import mpl_toolkits.mplot3d.axes3d as p3
from itertools import cycle
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, RandomizedPCA, NMF
from sklearn.preprocessing import scale, Normalizer, Binarizer 
from sklearn.pipeline import Pipeline, make_pipeline
from time import time
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist


from reader import load_data, create_feature_matrix, create_rank_matrix, rank_distance


def get_random_points(n, order):
	'''randomly gets n/order indexes from n, which is an array
	representing 3 classes'''
	clen = int(n/3)
	idx1 = np.linspace(0, clen-1, clen, dtype=int)
	idx2 = np.linspace(clen, clen*2-1, clen*2, dtype=int)
	idx3 = np.linspace(clen*2, clen*3-1, clen*3, dtype=int)

	num =  int(clen/order)        # set the number to select here.
	lstr = random.sample(idx1, num)
	lstr = lstr + random.sample(idx2, num)
	lstr = lstr + random.sample(idx3, num)
	print "len lstr", len(lstr)
	return lstr

def plot_agglomerative(X, n_labels, y):
	# Compute clustering
	print("Compute unstructured hierarchical clustering...")
	st = time()
	ward = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
	elapsed_time = time() - st
	label = ward.labels_
	# print("Elapsed time: %.2fs" % elapsed_time)
	# print("Number of points: %i" % label.size)

	# Plot result
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.view_init(7, -80)
	for l in np.unique(label):
	    ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
	              'o', color=plt.cm.jet(np.float(l) / np.max(label + 1)))
	plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)
	plt.savefig("hk.png")
	plot_agg_dendrogram(ward, labels=labels)

def plot_agg_dendrogram(model, **kwargs):
	fig = plt.figure()
	print("entered plot_dendrogram")
	# Children of hierarchical clustering
	children = model.children_

	# Distances between each pair of children
	# Since we don't have this information, we can use a uniform one for plotting
	distance = np.arange(children.shape[0])

	# The number of observations contained in each cluster level
	no_of_observations = np.arange(2, children.shape[0]+2)

	# Create linkage matrix and then plot the dendrogram
	linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

	# Plot the corresponding dendrogram
	dendrogram(linkage_matrix, **kwargs)

	plt.title("Dendrogram of Court Ruling Labels")
	plt.savefig("dend.png")


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def hierarchical(X, lb=None, dist_m=False):
	'''using scipy
	'''
	# generate the linkage matrix
	print("entered hierarchical")
	if dist_m == True: # X is the symmetric distance matrix
		# convert the redundant n*n square matrix form into a condensed nC2 array
		y = ssd.squareform(X) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
		Z = linkage(y, 'weighted', metric=rank_distance)
	else: # X is an (n, m) feature matrix
		Z = linkage(X, 'ward')

	c, coph_dists = cophenet(Z, pdist(X))
	print("coph dist" + str(c))

	plt.figure(figsize=(25, 10))
	plt.title('Hierarchical Clustering Dendrogram of Court Ruling Labels')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dendrogram(Z,
	    leaf_rotation=90.,  # rotates the x axis labels
	    leaf_font_size=3.,  # font size for the x axis labels
	    labels=lb,
	    show_contracted=True,
	    color_threshold=30
	)
	plt.savefig("scipyhierdend.png", format="png", dpi=500)


	plt.title('Hierarchical Clustering Dendrogram (truncated)')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dendrogram(
	    Z,
	    truncate_mode='lastp',  # show only the last p merged clusters
	    p=12,  # show only the last p merged clusters
	    show_leaf_counts=False,  # otherwise numbers in brackets are counts
	    leaf_rotation=90.,
	    leaf_font_size=8.,
	    show_contracted=True,  # to get a distribution impression in truncated branches
	)
	plt.savefig("scipyhierdend-trunc.png", format="png", dpi=1000)

	plt.figure(figsize=(30,30))
	fancy_dendrogram(
	    Z,
	    truncate_mode='lastp',
	    p=30,
	    leaf_rotation=90.,
	    leaf_font_size=8.,
	    show_contracted=True,
	    annotate_above=40,
	    max_d=170,
	)
	plt.savefig("fancy_dendrogram.png", format="png", dpi=1000)


if __name__ == '__main__':
	labels, vocab = load_data()
	n_labels = len(labels)
	print "n_labels", n_labels
	# docs_v = create_feature_matrix(labels, vocab)
	# docs_r = create_rank_matrix(docs_v)
	# plot_agglomerative(docs_v, 3, labels)
	# hierarchical(docs_r, True)
	# hierarchical(docs_v)
