#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import numpy as np
from cluster import *
from sklearn import manifold
from plot_utils import *

def plot_rho_delta(rho, delta):
	'''
	Plot scatter diagram for rho-delta points

	Args:
		rho   : rho list
		delta : delta list
	'''
	logger.info("PLOT: rho-delta plot")
	plot_scatter_diagram(0, rho[1:], delta[1:], x_label='rho', y_label='delta', title='Decision Graph')
	plt.savefig('Decision Graph.jpg')


def plot_cluster(cluster):
	'''
	Plot scatter diagram for final points that using multi-dimensional scaling for data

	Args:
		cluster : DensityPeakCluster object
	'''
	logger.info("PLOT: cluster result, start multi-dimensional scaling")
	dp = np.zeros((cluster.max_id, cluster.max_id), dtype = np.float32)
	cls = []
	for i in xrange(1, cluster.max_id):
		for j in xrange(i + 1, cluster.max_id + 1):
			dp[i - 1, j - 1] = cluster.distances[(i, j)]
			dp[j - 1, i - 1] = cluster.distances[(i, j)]
		cls.append(cluster.cluster[i])
	cls.append(cluster.cluster[cluster.max_id])
	cls = np.array(cls, dtype = np.float32)
	fo = open(r'./tmp.txt', 'w')
	fo.write('\n'.join(map(str, cls)))
	fo.close()
	#seed = np.random.RandomState(seed=3)
	mds = manifold.MDS(max_iter=200, eps=1e-4, n_init=1,dissimilarity='precomputed')
	dp_mds = mds.fit_transform(dp.astype(np.float64))
	logger.info("PLOT: end mds, start plot")
	plot_scatter_diagram(1, dp_mds[:, 0], dp_mds[:, 1], title='2D Nonclassical Multidimensional Scaling', style_list = cls)
	plt.savefig("2D Nonclassical Multidimensional Scaling.jpg")


def plot_rhodelta_rho(rho, delta):
	'''
	Plot scatter diagram for rho*delta_rho points

	Args:
		rho   : rho list
		delta : delta list
	'''
	logger.info("PLOT: rho*delta_rho plot")
	y=rho*delta
	r_index=np.argsort(-y)
	x=np.zeros(y.shape[0])
	idx=0
	for r in r_index:
	    x[r]=idx
	    idx+=1
	plt.figure(2)
	plt.clf()
	plt.scatter(x,y)
	plt.xlabel('sorted rho')
	plt.ylabel('rho*delta')
	plt.title("Decision Graph RhoDelta-Rho")
	plt.show()
	plt.savefig('Decision Graph RhoDelta-Rho.jpg')


if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	dpcluster = DensityPeakCluster()
	# dpcluster.local_density(load_paperdata, './example_distances.dat')
	# plot_rho_delta(rho, delta)   #plot to choose the threthold
	rho, delta, nneigh = dpcluster.cluster(load_paperdata, './data/data_in_paper/example_distances.dat', 20, 0.1)
	logger.info(str(len(dpcluster.ccenter)) + ' center as below')
	for idx, center in dpcluster.ccenter.items():
		logger.info('%d %f %f' %(idx, rho[center], delta[center]))
	plot_cluster(dpcluster)