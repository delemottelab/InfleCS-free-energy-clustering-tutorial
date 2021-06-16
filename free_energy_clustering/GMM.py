import sys
import math
import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture():
	
	def __init__(self,n_components=2, convergence_tol=1e-6, verbose=False):
		self.n_components_ = n_components
		self.weights_ = np.ones(n_components)/float(n_components)
		self.means_ = np.zeros(n_components)
		self.covariances_ = [np.zeros((n_components,n_components))]*n_components
		self.tol_ = convergence_tol
		self.data_weights_ = None
		self.verbose_ = verbose
		return
	
	def fit(self, x, data_weights=None):
		"""
		Fit GMM to points in x with EM.
		:param data_weights: Weights of each data point.
		"""

		if data_weights is not None:
			x = x[data_weights>0]
			data_weights = data_weights[data_weights>0]
			data_weights = data_weights/np.sum(data_weights)
			data_weights = data_weights * data_weights.shape[0]
		
		self.data_weights_ = data_weights
		while True:
			prev_loglikelihood = np.inf
			loglikelihood = 0
			self._initialize_parameters(x)

			while(np.abs(prev_loglikelihood-loglikelihood) > self.tol_):

				gamma = self._expectation(x, self.data_weights_)
				self._maximization(x, gamma)

				prev_loglikelihood= loglikelihood
				loglikelihood = self.loglikelihood(x, self.data_weights_)

			break
		return self

	def predict(self,x):
		gamma = self._expectation(x)
		labels = np.argmax(gamma,axis=0)
		return labels
	
	def _initialize_parameters(self,x):
		"""
		Initialize component means and covariances
		"""
		n_points = x.shape[0]
		inds = np.random.randint(n_points,size=self.n_components_)
		
		# Initialize means
		self.means_ = x[inds,:]
		# Initialize covariances
		tmp_cov = np.cov(x.T)
		for i_component in range(self.n_components_):
			self.covariances_[i_component] = tmp_cov
		return

	def _expectation(self,x, data_weights=None):
		"""
		Perform expecation step
		"""
		n_points = x.shape[0]
		gamma = np.zeros((self.n_components_,n_points))

		for i_component in range(self.n_components_):

			normal_density = multivariate_normal.pdf(x, mean=self.means_[i_component], cov=self.covariances_[i_component])
			gamma[i_component, :] = self.weights_[i_component]*normal_density
		
		gamma /= np.sum(gamma,axis=0)			
		
		if data_weights is not None:
			gamma = np.multiply(gamma, data_weights)
		
		return gamma
	
	def _maximization(self,x, gamma):
		"""
		Update parameters with maximization step
		"""
		self._update_weights(x, gamma)
		self._update_means(x, gamma)
		self._update_covariances(x, gamma)
		return
	
	def _update_weights(self,x, gamma):
		"""
		Update each component amplitude.
		"""
		
		self.weights_ = np.sum(gamma,axis=1)
		
		# Normalize Cat-distibution
		self.weights_ /= np.sum(self.weights_)
		return


	def _update_means(self,x, gamma):
		"""
		Update each component mean.
		"""
		Nk = np.sum(gamma,axis=1)
		for i_component in range(self.n_components_):
			self.means_[i_component, :] = np.dot(x.T,gamma[i_component])/Nk[i_component]

		return
	
	def _update_covariances(self, x, gamma):
		"""
		Update each component covariance
		"""
		n_dims = x.shape[1]

		Nk = np.sum(gamma, axis=1)
		for i_component in range(self.n_components_):
			y = x - self.means_[i_component]
			y2 = np.multiply(gamma[i_component,:,np.newaxis],y).T
			self.covariances_[i_component] = y2.dot(y)/Nk[i_component] + 1e-9*np.eye(n_dims)

		return
		
	def density(self, x):
		"""
		Compute GMM density at given points, x.
		"""
		n_points = x.shape[0]
		n_dims = x.shape[1]

		density = np.zeros(n_points)
		for i_component in range(self.n_components_):
			normal_density = multivariate_normal.pdf(x, mean=self.means_[i_component], cov=self.covariances_[i_component])
			density += self.weights_[i_component]*normal_density
		
		return density
	
	def loglikelihood(self, x, data_weights=None):
		"""
		Compute log-likelihood. Support data weights.
		"""
		density = self.density(x)
		density[density<1e-15] = 1e-15
		if data_weights is None:
			log_density = np.log(density)
		else:
			log_density = np.multiply(np.log(density), data_weights)
		return np.mean(log_density)

	def bic(self, x, data_weights=None):
		"""
        Compute BIC score. Support data weights.
        """
		n_points, n_dims = x.shape
		n_params = (1 + n_dims + n_dims * (n_dims + 1) / 2.0) * self.n_components_
		loglikelihood = n_points * self.loglikelihood(x, data_weights=data_weights)
		return -2.0 * loglikelihood + n_params * math.log(n_points)

	def aic(self, x, data_weights=None):
		"""
		Compute BIC score. Support data weights.
		"""
		n_points, n_dims = x.shape
		n_params = (1 + n_dims + n_dims * (n_dims + 1) / 2.0) * self.n_components_
		loglikelihood = n_points * self.loglikelihood(x, data_weights=data_weights)
		return -2.0 * loglikelihood + 2.0 * n_params

	def sample(self, n_points):
		"""
        Sample points from the density model.
        :param n_points:
        :return:
        """
		n_dims = self.means_.shape[1]
		sampled_points = np.zeros((n_points, n_dims))
		prob_component = np.cumsum(self.weights_)
		r = np.random.uniform(size=n_points)

		is_point_sampled = np.zeros((n_points), dtype=int)

		for i_point in range(n_points):
			for i_component in range(self.n_components_):
				if r[i_point] <= prob_component[i_component]:
					sampled_points[i_point, :] = np.random.multivariate_normal(self.means_[i_component],
																			   self.covariances_[i_component], 1)
					is_point_sampled[i_point] = 1
					break
			if is_point_sampled[i_point] ==0:
				print('Warning: Did not sample point: '+str(r[i_point])+' '+str(prob_component))
		return sampled_points
