import numpy as np
from scipy.optimize import fmin_cg
import GMM_FE.cluster_density as cluster
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

class LandscapeClustering():

	def __init__(self, ensemble_of_GMMs=False):
		self.cluster_centers_ = None
		self.labels_ = None
		self.ensemble_of_GMMs = ensemble_of_GMMs
		self.clusterer_ = None
		return
	
	def get_cluster_representative(self, x, labels, free_energies):
		"""
		Get one point in each cluster that has minimum FE in that cluster
		"""
		n_clusters = int(np.max(labels) + 1)
		n_points = x.shape[0]

		print('Cluster labels: '+str(np.unique(labels)))

		min_FE_inds = np.zeros(n_clusters-1)
		all_inds = np.arange(n_points)
		mask = np.ones(n_clusters-1,dtype=bool)
		for i_cluster in range(1,n_clusters):
			cluster_inds = all_inds[labels == i_cluster]
			if cluster_inds.shape[0] > 0:
				min_FE_inds[i_cluster-1] = cluster_inds[np.argmin(free_energies[cluster_inds])]
			else:
				min_FE_inds[i_cluster-1] = np.nan
				mask[i_cluster-1]=False
				print('No point in cluster '+str(i_cluster))
		
		self.cluster_centers_ = min_FE_inds[mask].astype(int)
		return self.cluster_centers_

	def minimization_fun(self, x, *args):
		density_model,_ = args
		score = -density_model.density(x[np.newaxis,:])
		return score

	def minimization_grad(self, x, *args):
		density_model,_ = args
		grad,_ = self._compute_gradients(density_model, x[np.newaxis,:])
		return -np.ravel(grad)

	def assign_transition_points(self, cluster_indices, points, density_model):
		"""
		Assign cluster indices to transition points by maximizing density towards local maximum and use this to assign
		the cluster index.
		:return:
		"""
		print("Assigning cluster indices to non-core cluster points.")
		cl_inds_final = np.copy(cluster_indices)
		transition_point_inds = np.where(cluster_indices==0)[0]
		n_assigned = np.sum(cluster_indices>0)
		assigned_points = points[0:n_assigned,:]

		# Sort points from higher to lower density
		density_all = density_model.density(points)
		#density_all[transition_point_inds] = -1
		densities_trans_points = density_all[transition_point_inds]

		# Sort transition points in decending density order (assign cluster index to highest density points first)
		sort_inds = np.argsort(-densities_trans_points)
		transition_point_inds = transition_point_inds[sort_inds]

		counter = 0
		for ind in transition_point_inds:

			point = points[ind]
			# Extract assigned points
			assigned_inds = np.where(cl_inds_final>0)[0]
			assigned_points = points[assigned_inds,:]
			distances = cdist(point[np.newaxis,:],assigned_points)

			# Find closest assigned point. Use its cluster index on the current unassigned point.
			closest_point = np.argmin(distances[0,:])
			cl_inds_final[ind] = cl_inds_final[assigned_inds[closest_point]]

			n_assigned += 1
			counter += 1
		return cl_inds_final

	def _compute_gradients(self, density_model, points):
		n_points = points.shape[0]
		n_dims = points.shape[1]
		n_components = density_model.n_components_

		means = density_model.means_
		covs = density_model.covariances_
		weights = density_model.weights_

		gradients = np.zeros((n_points, n_dims))

		inv_covs = [np.zeros((n_dims, n_dims))] * n_components
		for i_component in range(n_components):
			inv_covs[i_component] = np.linalg.inv(covs[i_component])

			devs = points - means[i_component]
			exp_deriv = -devs.dot(inv_covs[i_component])
			for i_point in range(n_points):
				gradients[i_point, :] += weights[i_component] * exp_deriv[i_point, :] * multivariate_normal.pdf(
					points[i_point, :], mean=means[i_component], cov=covs[i_component])

		return gradients, inv_covs

	def _compute_GMM_Hessian(self, density_model, x, inv_covs):
		n_dims = x.shape[0]
		n_components = density_model.n_components_

		means = density_model.means_
		covs = density_model.covariances_
		weights = density_model.weights_

		hessian = np.zeros((n_dims, n_dims))

		for i_component in range(n_components):
			devs = x - means[i_component]
			exp_deriv = -devs.dot(inv_covs[i_component])

			# Compute Hessian at current point
			for i_dim in range(n_dims):
				for j_dim in range(n_dims):
					post_weight = weights[i_component] * multivariate_normal.pdf(x, mean=means[i_component],
																				 cov=covs[i_component])
					hessian[i_dim, j_dim] += post_weight * (
								-inv_covs[i_component][i_dim, j_dim] + exp_deriv[i_dim] * exp_deriv[j_dim])

		return hessian

	def _Hessian_def(self, density_model, points):
		"""
		Compute the Hessian in every point to check whether they belong to a 
		free energy minimum or not.
		"""
		n_points = points.shape[0]
		n_dims = points.shape[1]

		if self.ensemble_of_GMMs:
			n_models = density_model.n_models_

		is_FE_min = [False] * n_points
		
		# Compute all inverse covariances
		if self.ensemble_of_GMMs:
			all_inv_covs = [0]*n_models
			n_components = [0]*n_models
			for i_model in range(n_models):

				n_components = density_model.GMM_list_[i_model].n_components_

				inv_covs = [np.zeros((n_dims, n_dims))] * n_components
				for i_component in range(n_components):
					inv_covs[i_component] = np.linalg.inv(density_model.GMM_list_[i_model].covariances_[i_component])
				all_inv_covs[i_model] = inv_covs
		else:
			n_components = density_model.n_components_
			inv_covs = [np.zeros((n_dims, n_dims))] * n_components
			for i_component in range(n_components):
				inv_covs[i_component] = np.linalg.inv(density_model.covariances_[i_component])

		# Computing Hessian to determine whether point belongs to FE min or not
		# neg definite => True, else => False
		print('Computing Hessians.')
		for i_point, x in enumerate(points):
			if self.ensemble_of_GMMs:
				hessian = np.zeros((n_dims,n_dims))
				for i_model in range(n_models):
					if density_model.model_weights_[i_model] > 0:
						hessian += density_model.model_weights_[i_model]*self._compute_GMM_Hessian(density_model.GMM_list_[i_model],
																							   x, all_inv_covs[i_model])
			else:
				hessian = self._compute_GMM_Hessian(density_model, x, inv_covs)
			
			# Compute Hessian eigenvalues
			eigvals = np.linalg.eigvals(hessian)
						
			# Check if Hessian is negative definite, the point is at a free energy minimum
			if eigvals.max() < 0.0:
				is_FE_min[i_point] = True
		
		return is_FE_min

	def cluster(self, density_models, points, eval_points=None):
		# Indicate whether points are at free energy minimum or not
		is_FE_min = self._Hessian_def(density_models, points)
		self.grid_points_=points
		# Cluster free energy landscape
		self.clusterer_ = cluster.ClusterDensity(points, eval_points)
		self.labels_ = self.clusterer_.cluster_data(is_FE_min)
		return self.labels_, is_FE_min
		
