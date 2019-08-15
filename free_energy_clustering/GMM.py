import sys
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
	
	def fit(self, x, data_weights=None, bias_factors=None):
		"""
		Fit GMM to points in x with EM.
		:param data_weights: Weights of each data point. These can be estimated given input bias_factor.
		:param bias_factors: Bias factors in exponent. b(x) = -bias_factor*free_energy.
		"""

		if data_weights is not None:
			x = x[data_weights>0]
			data_weights = data_weights[data_weights>0]

		if bias_factors is not None and data_weights is None:
			data_weights = np.ones(x.shape[0])
		
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

			if bias_factors is not None:
				prev_data_weights = np.copy(self.data_weights_)
				self.compute_data_weights(x, bias_factors)

				change = np.linalg.norm(prev_data_weights/prev_data_weights.sum() - self.data_weights_/self.data_weights_.sum())
				if self.verbose_:
					sys.stdout.write("\r" + 'Data weight change: '+str(change))
					sys.stdout.flush()
				if change < 5e-5:
					break
			else:
				break

		if bias_factors is not None and self.verbose_:
			print()
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

	def _reweight_normal_density(self, density, data_weights, n_dims):
		"""
		Reweight the normal probability density using the data weights.
		:param density:
		:param data_weights:
		:return:
		"""

		# Reweight density
		new_density = np.power(density,data_weights)
		
		# Normalize density
		new_density = np.multiply(new_density, np.power(data_weights,float(n_dims)/2.0))

		return new_density

	def _expectation(self,x, data_weights=None):
		"""
		Perform expecation step
		"""
		n_points = x.shape[0]
		n_dims = x.shape[1]
		gamma = np.zeros((self.n_components_,n_points))

		for i_component in range(self.n_components_):

			normal_density = multivariate_normal.pdf(x, mean=self.means_[i_component], cov=self.covariances_[i_component])
			gamma[i_component,:] = self.weights_[i_component]*normal_density
		
		gamma /= np.sum(gamma,axis=0)			
		
		if data_weights is not None:
			gamma = np.multiply(gamma,data_weights)	
		
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
			self.means_[i_component,:] = np.dot(x.T,gamma[i_component])/Nk[i_component]

		return
	
	def _update_covariances(self,x, gamma):
		"""
		Update each component covariance
		"""
		n_dims = x.shape[1]

		Nk = np.sum(gamma,axis=1)
		for i_component in range(self.n_components_):
			y = x - self.means_[i_component]
			y2 = np.multiply(gamma[i_component,:,np.newaxis],y).T
			self.covariances_[i_component] = y2.dot(y)/Nk[i_component] + 1e-9*np.eye(n_dims)

		return

	def compute_data_weights(self, x, bias_factors):
		"""
		Update the data weights using the current estimated density.
		:param x: data
		:param bias_factors: Bias factors in exponent. b(x) = bias_factor(x)*free_energy(x).
		Option two: without bias factors: The weights are obtained from the fraction
		"density(x) in ensemble k divided by density(x) in ensemble 0"
		:return:
		"""
		using_bias_factors = False
		if using_bias_factors:
			# Compute current free energy estimate
			density = self.density(x)
			density[density<1e-15]=1e-15
			free_energy = -np.log(density)

			# Reweight points
			self.data_weights_ = np.exp(np.multiply(bias_factors,free_energy))
		else:
			densities_biased = self.density(x, self.data_weights_)
			densities_unbiased = self.density(x)
			self.data_weights_ = densities_biased/densities_unbiased
			print('Size data weights: '+str(self.data_weights_.shape))

		self.data_weights_[self.data_weights_<1e-8] = 1e-8
		if np.any(np.isnan(self.data_weights_)):
			print('Warning: NaN values in data weights!')

		return
		
	def density(self,x):
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
	
	def loglikelihood(self,x, data_weights=None):
		"""
		Compute log-likelihood. Possibility of data weights.
		"""
		density = self.density(x)
		density[density<1e-15] = 1e-15
		if data_weights is None:
			log_density = np.log(density)
		else:
			log_density = np.multiply(np.log(density),data_weights)
		return np.mean(log_density)

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
