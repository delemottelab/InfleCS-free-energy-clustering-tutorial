import numpy as np
from sklearn import datasets
from free_energy_clustering.GMM import GaussianMixture

class Blobs(GaussianMixture):
	def __init__(self,n_components=3,n_dims=2,noise=0):
		GaussianMixture.__init__(self, n_components=n_components)
		self.labels_ = None
		self.data_ = None
		self.n_features_ = n_dims
		self.noise_level_ = noise
		self.name = 'blobs'
		return

	def sample(self, n_points):
		self.data_, self.labels_ = datasets.make_blobs(n_samples=n_points, n_features=self.n_features_)
		print(self.data_.shape)

		self.set_density()

		# Sample noise uniformly over space
		n_noise_points = int(self.noise_level_ * self.data_.shape[0])
		data_noise = np.random.uniform(self.data_.min(axis=0), self.data_.max(axis=0),
									   size=(n_noise_points, self.data_.shape[1]))

		self.data_[0:n_noise_points] = data_noise
		return self.data_

	def set_density(self):
		
		unique_labels = np.unique(self.labels_)

		self.weights_ = np.zeros(self.n_components_)
		self.means_ = np.zeros((self.n_components_,self.n_features_))
		self.covariances_ = [np.zeros((self.n_features_,self.n_features_))]*self.n_components_
		for label in unique_labels:
			self.weights_[label] = np.mean(self.labels_==label)
			self.means_[label] = np.mean(self.data_[self.labels_==label],axis=0)
			self.covariances_[label] = np.cov(self.data_[self.labels_==label].T)
		return

	def assign_cluster_labels(self, x):
		return self.predict(x)

