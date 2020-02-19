import numpy as np
from free_energy_clustering.GMM import GaussianMixture

class MultipleGMMs(GaussianMixture):
	def __init__(self):
		data = np.zeros((3,2))
		GaussianMixture.__init__(self, n_components=10)
		
		self.name = 'mGMMs'
		self.n_dims_ = 2
		self._set_parameters()
		return

	def sample_multi_GMM(self, n_points):
		"""
		Sample from stacked GMMs.
		"""
		sampled_points = np.zeros((n_points, self.n_dims_))
		prob_model = np.cumsum(self.model_weights_)

		r = np.random.uniform(size=n_points)
		is_point_sampled = np.zeros((n_points), dtype=int)

		for i_point in range(n_points):
			for i_model in range(self.n_models_):
				if r[i_point] <= prob_model[i_model]:
					is_point_sampled[i_model] += 1
					sampled_points[i_point,:] = self.GMM_list_[i_model].sample(1)
					break
		print('Sampled: '+str(is_point_sampled.sum())+'/'+str(n_points))
		return sampled_points

	def _set_cov(self, x11,x12,x22):
		tmp_cov = np.zeros((self.n_dims_, self.n_dims_))

		tmp_cov[0, 0] = x11
		tmp_cov[0, 1] = x12
		tmp_cov[1, 0] = x12
		tmp_cov[1, 1] = x22
		return tmp_cov

	def _set_GMM1(self):
		n_components = 4
		means = np.asarray([np.asarray([0.5,0.27]), np.asarray([0.5, 0.27]), np.asarray([0.5, 0.27]),
							np.asarray([0.5, 0.27])])

		covs = [np.zeros((2, 2))] * n_components

		covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
		covs[1] = self._set_cov(0.001, 0.0009, 0.001)
		covs[2] = self._set_cov(0.002, 0.001, 0.002)
		covs[3] = self._set_cov(0.003, 0.0008, 0.003)

		weights = np.asarray([0.5, 0.3,0.3,0.3])
		weights /= weights.sum()
		return means, covs, weights

	def _set_GMM2(self):

		n_components = 3
		means = np.asarray([np.asarray([0.45, 0.5]), np.asarray([0.45, 0.5]), np.asarray([0.45, 0.5])])

		covs = [np.zeros((2, 2))] * n_components

		covs[0] = self._set_cov(0.01, 0.0, 0.0001)
		covs[1] = self._set_cov(0.0003, 0.0003, 0.015)
		covs[2] = self._set_cov(0.0012, 0.00, 0.002)

		weights = np.asarray([0.5, 0.2, 0.3])
		weights /= weights.sum()
		return means, covs, weights

	def _set_GMM3(self):
		n_components = 3
		means = np.asarray(
			[np.asarray([0.05, 0.8]), np.asarray([0.05, 0.8]), np.asarray([0.05, 0.8])])

		covs = [np.zeros((2, 2))] * n_components

		covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
		covs[1] = self._set_cov(0.001, 0.0009, 0.001)
		covs[2] = self._set_cov(0.002, 0.001, 0.002)

		weights = np.ones(n_components)
		weights /= weights.sum()
		return means, covs, weights

	def _set_GMM12(self):
		n_components = 9
		means = np.asarray([ np.asarray([0.8,0.35]), np.asarray([0.45,0.5]), np.asarray([0.2,0.6]),
								   np.asarray([0.05,0.8]), np.asarray([0.5,0.27]), np.asarray([0.5,0.27]),
								   np.asarray([0.5, 0.27]), np.asarray([0.5, 0.4]), np.asarray([0.8,0.5])])

		covs = [np.zeros((2,2))]*n_components

		covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
		covs[1] = self._set_cov(0.001, 0.0009, 0.001)
		covs[2] = self._set_cov(0.002, 0.001, 0.002)
		covs[3] = self._set_cov(0.003, 0.0008, 0.003)
		covs[4] = self._set_cov(0.0012, 0.0005, 0.0012)
		covs[5] = self._set_cov(0.01, 0.0, 0.0015)
		covs[6] = self._set_cov(0.005, 0.001, 0.02)
		covs[7] = self._set_cov(0.002, -0.0001, 0.002)
		covs[8] = self._set_cov(0.001, 0.0009, 0.001)

		weights = np.asarray([0.15,0.1,0.5,0.25,0.1,0.05,0.1,0.05,0.4])
		weights /= weights.sum()
		return means, covs, weights

	def _set_GMM22(self):

		n_components = 3
		means = np.asarray([ np.asarray([0.6,0.35]), np.asarray([0.45,0.5]), np.asarray([0.19,0.62])])
		
		covs = [np.zeros((2,2))]*n_components

		covs[0] = self._set_cov(0.003, 0.0008, 0.003)
		covs[1] = self._set_cov(0.005, 0.0, 0.0015)
		covs[2] = self._set_cov(0.0012, 0.00, 0.002)
		
		weights = np.asarray([0.5,0.2,0.3])
		weights /= weights.sum()
		return means, covs, weights


	def _set_GMM32(self):
		n_components = 5
		means = np.asarray([ np.asarray([0.05,0.8]),np.asarray([0.05,0.8]), np.asarray([0.52,0.25]), np.asarray([0.52,0.27]), np.asarray([0.45, 0.5])])
		
		covs = [np.zeros((2,2))]*n_components
		
		covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
		covs[1] = self._set_cov(0.001, 0.0009, 0.001)
		covs[2] = self._set_cov(0.002, 0.001, 0.002)
		covs[3] = self._set_cov(0.003, 0.0008, 0.003)
		covs[4] = self._set_cov(0.0012, 0.0005, 0.0012)

		weights = np.asarray([0.15,0.25,0.10,0.3,0.2])
		weights /= weights.sum()
		return means, covs, weights

	def assign_cluster_labels(self,x):
		gamma = self._expectation(x)
		labels = np.argmax(gamma, axis=0)+1
		return labels

	def _expectation(self, x):
		n_points = x.shape[0]
		gamma = np.zeros((self.n_models_, n_points))

		for i_model in range(self.n_models_):
			gamma[i_model, :] = self.model_weights_[i_model] *self.GMM_list_[i_model].density(x)
		gamma /= np.sum(gamma, axis=0)
		return gamma

	def _set_parameters(self):
		n_components = 10
		means = np.asarray([np.asarray([0.5, 0.27]), np.asarray([0.5, 0.27]), np.asarray([0.5, 0.27]),
							np.asarray([0.5, 0.27]), np.asarray([0.40, 0.5]), np.asarray([0.4, 0.5]),
						   np.asarray([0.4, 0.5]),np.asarray([0.05, 0.8]), np.asarray([0.05, 0.8]),
							np.asarray([0.05, 0.8])])

		covs = [np.zeros((2, 2))] * n_components

		covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
		covs[1] = self._set_cov(0.001, 0.0009, 0.001)
		covs[2] = self._set_cov(0.002, 0.001, 0.002)
		covs[3] = self._set_cov(0.003, 0.0008, 0.003)

		weights1 = np.asarray([0.5, 0.3, 0.3, 0.3])
		weights1 /= weights1.sum()

		covs[4] = self._set_cov(0.01, 0.0, 0.0001)
		covs[5] = self._set_cov(0.0003, 0.0000, 0.0003)
		covs[6] = self._set_cov(0.0012, 0.00, 0.002)

		weights2 = np.asarray([0.5, 0.2, 0.3])
		weights2 /= weights2.sum()

		covs[7] = self._set_cov(0.0021, 0.0005, 0.002)
		covs[8] = self._set_cov(0.001, 0.0009, 0.001)
		covs[9] = self._set_cov(0.002, 0.001, 0.002)

		weights3 = np.ones(3)
		weights3 /= weights3.sum()

		weights = np.ravel(np.concatenate((0.25*weights1, 0.5*weights2, 0.25*weights3)))

		self.means_ = means
		self.covariances_ = covs
		self.weights_ = weights

		self.GMM_list_ = []
		self.GMM_list_.append(GaussianMixture(n_components=4))
		self.GMM_list_[-1].means_ = means[0:4,:]
		self.GMM_list_[-1].covariances_ = covs[0:4]
		self.GMM_list_[-1].weights_ = weights1

		self.GMM_list_.append(GaussianMixture(n_components=3))
		self.GMM_list_[-1].means_ = means[4:7,:]
		self.GMM_list_[-1].covariances_ = covs[4:7]
		self.GMM_list_[-1].weights_ = weights1

		self.GMM_list_.append(GaussianMixture(n_components=3))
		self.GMM_list_[-1].means_ = means[7::, :]
		self.GMM_list_[-1].covariances_ = covs[7::]
		self.GMM_list_[-1].weights_ = weights1

		self.model_weights_ = np.asarray([0.25,0.5,0.25])
		self.n_models_ = 3
		return
