import sys
import numpy as np
import toy_models as tm
import free_energy_clustering as GMM_FE
from toy_models import Kmeans_cluster as kmc
from toy_models import spectral_cluster as sc
from toy_models import agglomerative_ward_cluster as awc

from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.cluster import KMeans

class MethodEvaluator(object):

	def __init__(self, toy_model='GMM_2D', x_lims=None, n_grids=30, convergence_tol=1e-4,verbose=False, presampled_data=None, n_features=None, noise=0, n_components=3):

		if toy_model == 'GMM_2D':
			self.toy_model_ = tm.GMM2D()
		elif toy_model == 'mGMMs':
			self.toy_model_ = tm.MultipleGMMs()
		elif toy_model == 'blobs':
			self.toy_model_ = tm.Blobs(n_components=n_components, n_dims=n_features, noise=noise)
		elif toy_model == 'digits':
			self.toy_model_ = tm.Digits()
		elif toy_model == 'nonlinear_GMM_2D':
			self.toy_model_ = tm.GMM2dNonlinear()
		else:
			print('Toy model: '+str(toy_model)+' does not exist')
			sys.exit(0)
		
		self.cluster_score_ami_kmeans_ = None
		self.cluster_score_ami_AW_ = None
		self.cluster_score_ami_spectral_ = None
		self.cluster_score_ami_density_peaks_ = None
		self.cluster_score_ami_GMM_ = None
		self.cluster_score_ami_GMM_FE_min_ = None

		self.cluster_score_fm_kmeans_ = None
		self.cluster_score_fm_AW_ = None
		self.cluster_score_fm_spectral_ = None
		self.cluster_score_fm_density_peaks_ = None
		self.cluster_score_fm_GMM_ = None
		self.cluster_score_fm_GMM_FE_min_ = None

		self.cluster_score_vm_kmeans_ = None
		self.cluster_score_vm_AW_ = None
		self.cluster_score_vm_spectral_ = None
		self.cluster_score_vm_density_peaks_ = None
		self.cluster_score_vm_DPA_ = None
		self.cluster_score_vm_HDBSCAN_ = None
		self.cluster_score_vm_SDC_ = None
		self.cluster_score_vm_GMM_ = None
		self.cluster_score_vm_GMM_FE_min_ = None

		self.convergence_tol_ = convergence_tol

		self.x_lims_ = x_lims
		self.n_grids_ = n_grids

		self.presampled_data = presampled_data

		self.true_FE_ = None
		self.true_density_ = None
		self.true_labels_ = None
		self.test_set_ = None
		self.min_FE_ = None
		self.verbose_ = verbose

		self.set_true_free_energy()
		return

	def set_true_free_energy(self):
		"""
		Create a free energy object that contains the true free energy and density on the given grid.
		:return:
		"""
		# Create grid and evaluate density on it
		print('Setting true model.')
		self.test_set_ = self.toy_model_.sample(2000)
		self.true_FE_ = GMM_FE.FreeEnergyClustering(self.test_set_, x_lims=self.x_lims_, n_grids=self.n_grids_,verbose=False,
										  convergence_tol=self.convergence_tol_)
		self.true_FE_.density_est_ = self.toy_model_

		coords, self.true_density_ = self.true_FE_._density_landscape(self.toy_model_)

		# Compute true free energy
		FE_landscape = self.true_FE_._free_energy(self.true_density_)
		self.min_FE_= np.min(FE_landscape)
		FE_landscape = FE_landscape - self.min_FE_

		# Set true free energy
		self.true_FE_.coords_ = coords
		self.true_FE_.FE_landscape_ = FE_landscape

		if hasattr(self.toy_model_,"assign_cluster_labels"):
			self.true_labels_ = self.toy_model_.assign_cluster_labels(self.test_set_)
		else:
			self.true_labels_, _ = self.true_FE_.cluster(coords, np.zeros(self.test_set_.shape[0]), self.test_set_)
		return

	def run_evaluation(self, n_runs=1, n_points=1000, n_iterations=1, min_n_components=2, max_n_components=25,
					   n_splits=3, save_data=False, file_label='',n_microstates=None, all_methods=True,
					   assign_transition_points=True):
		"""
		Run multiple free energy estimations and evaluate performance.
		:param n_runs:
		:return:
		"""

		if self.presampled_data is not None:
			sampled_data = self.presampled_data[0]
			true_clustering = self.presampled_data[1]
			n_runs = sampled_data.shape[0]

		self.cluster_score_ami_kmeans_ = np.zeros(n_runs)
		self.cluster_score_ami_AW_ = np.zeros(n_runs)
		self.cluster_score_ami_spectral_ = np.zeros(n_runs)
		self.cluster_score_ami_density_peaks_ = np.zeros(n_runs)
		self.cluster_score_ami_GMM_ = np.zeros(n_runs)
		self.cluster_score_ami_GMM_FE_min_ = np.zeros(n_runs)

		self.cluster_score_fm_kmeans_ = np.zeros(n_runs)
		self.cluster_score_fm_AW_ = np.zeros(n_runs)
		self.cluster_score_fm_spectral_ = np.zeros(n_runs)
		self.cluster_score_fm_density_peaks_ = np.zeros(n_runs)
		self.cluster_score_fm_GMM_ = np.zeros(n_runs)
		self.cluster_score_fm_GMM_FE_min_ = np.zeros(n_runs)

		self.cluster_score_vm_kmeans_ = np.zeros(n_runs)
		self.cluster_score_vm_AW_ = np.zeros(n_runs)
		self.cluster_score_vm_spectral_ = np.zeros(n_runs)
		self.cluster_score_vm_density_peaks_ = np.zeros(n_runs)
		self.cluster_score_vm_GMM_ = np.zeros(n_runs)
		self.cluster_score_vm_GMM_FE_min_ = np.zeros(n_runs)

		data = self.toy_model_.sample(3)

		# Create free energy estimators
		gmm_FE = GMM_FE.FreeEnergyClustering(data, min_n_components=min_n_components, max_n_components=max_n_components,
									 x_lims=self.x_lims_, n_grids=self.n_grids_, stack_landscapes=False,
									 n_splits=n_splits, n_iterations=n_iterations,convergence_tol=self.convergence_tol_,
									  verbose=self.verbose_)

		km = kmc.KMeansCluster(min_n_components, max_n_components)
		aw = awc.AWCluster(min_n_components, max_n_components)
		spectral = sc.SpectralCluster(min_n_components, max_n_components)

		all_data = []
		for i_run in range(n_runs):
			print("Run: "+str(i_run+1)+'/'+str(n_runs))

			if self.presampled_data is None:
				# Sample data
				data = self.toy_model_.sample(n_points)
			else:
				data = sampled_data[i_run]
			
			all_data.append(data)

			print('Shape data: ' + str(data.shape))

			# Set data in model and estimate GMM density
			gmm_FE.data_ = data
			coords, est_FE_landsc, FE_points = gmm_FE.landscape()

			# Get true cluster labels
			if self.presampled_data is None:
				if hasattr(self.toy_model_, "assign_cluster_labels"):
					self.true_labels_ = self.toy_model_.assign_cluster_labels(data)
				else:
					print('Setting true labels.')
					self.true_labels_, _ = self.true_FE_.cluster(data, np.zeros(data.shape[0]))
			else:
				self.true_labels_ = true_clustering[i_run]
			
			# Cluster data with different methods
			if n_microstates is None:
				self.FE_min_labels, _ = gmm_FE.cluster(data, FE_points, assign_transition_points=assign_transition_points)
			else:
				kmea = KMeans(n_clusters=n_microstates).fit(data[::2])
				microstate_centers = kmea.cluster_centers_
				self.FE_min_labels, _ = gmm_FE.cluster(microstate_centers, FE_points, data, assign_transition_points=assign_transition_points, unravel_grid=False)

			if all_methods:
				self.km_labels = km.cluster(data)
				self.aw_labels = aw.cluster(data)
				self.spectral_labels = spectral.cluster(data)

			# Score clustering using different scoring metrics
			# V-measure score
			self.cluster_score_vm_GMM_FE_min_[i_run] = self._score_clustering(self.FE_min_labels,'vm')
			print(self.cluster_score_vm_GMM_FE_min_[i_run])
			if all_methods:
				self.cluster_score_vm_GMM_[i_run] = self._score_clustering(gmm_FE.density_est_.predict(data),'vm')
				self.cluster_score_vm_kmeans_[i_run] = self._score_clustering(self.km_labels,'vm')
				self.cluster_score_vm_AW_[i_run] = self._score_clustering(self.aw_labels,'vm')
				self.cluster_score_vm_spectral_[i_run] = self._score_clustering(self.spectral_labels,'vm')

				# Adjusted MI
				self.cluster_score_ami_GMM_FE_min_[i_run] = self._score_clustering(self.FE_min_labels,'ami')
				self.cluster_score_ami_GMM_[i_run] = self._score_clustering(gmm_FE.density_est_.predict(data),'ami')
				self.cluster_score_ami_kmeans_[i_run] = self._score_clustering(self.km_labels,'ami')
				self.cluster_score_ami_AW_[i_run] = self._score_clustering(self.aw_labels,'ami')
				self.cluster_score_ami_spectral_[i_run] = self._score_clustering(self.spectral_labels,'ami')

				# Fowlkes Mallows
				self.cluster_score_fm_GMM_FE_min_[i_run] = self._score_clustering(self.FE_min_labels,'fm')
				self.cluster_score_fm_GMM_[i_run] = self._score_clustering(gmm_FE.density_est_.predict(data),'fm')
				self.cluster_score_fm_kmeans_[i_run] = self._score_clustering(self.km_labels,'fm')
				self.cluster_score_fm_AW_[i_run] = self._score_clustering(self.aw_labels,'fm')
				self.cluster_score_fm_spectral_[i_run] = self._score_clustering(self.spectral_labels,'fm')
		
		if save_data:
			if self.presampled_data is None:
				np.save('data_out/sampled_data_'+self.toy_model_.name+file_label+'.npy',all_data)

			if False:
				np.save('data_out/cluster_score_fm_FE_min_'+self.toy_model_.name+file_label+'.npy',self.cluster_score_fm_GMM_FE_min_)
				np.save('data_out/cluster_score_fm_GMM_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_fm_GMM_)
				np.save('data_out/cluster_score_fm_kmeans_' + self.toy_model_.name +file_label +'.npy', self.cluster_score_fm_kmeans_)
				np.save('data_out/cluster_score_fm_AW_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_fm_AW_)
				np.save('data_out/cluster_score_fm_spectral_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_fm_spectral_)

				np.save('data_out/cluster_score_ami_FE_min_'+self.toy_model_.name+file_label+'.npy',self.cluster_score_ami_GMM_FE_min_)
				np.save('data_out/cluster_score_ami_GMM_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_ami_GMM_)
				np.save('data_out/cluster_score_ami_kmeans_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_ami_kmeans_)
				np.save('data_out/cluster_score_ami_AW_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_ami_AW_)
				np.save('data_out/cluster_score_ami_spectral_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_ami_spectral_)

			np.save('data_out/cluster_score_vm_FE_min_'+self.toy_model_.name+file_label+'.npy',self.cluster_score_vm_GMM_FE_min_)
			if all_methods:
				np.save('data_out/cluster_score_vm_GMM_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_vm_GMM_)
				np.save('data_out/cluster_score_vm_kmeans_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_vm_kmeans_)
				np.save('data_out/cluster_score_vm_AW_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_vm_AW_)
				np.save('data_out/cluster_score_vm_spectral_' + self.toy_model_.name + file_label+'.npy', self.cluster_score_vm_spectral_)
		return

	def _score_clustering(self, labels,metric='vm'):
		# Score clustering compared to true model
		if metric=='fm':
			score = fowlkes_mallows_score(self.true_labels_, labels)
		elif metric=='ami':
			score = adjusted_mutual_info_score(self.true_labels_, labels)
		else:
			score = v_measure_score(self.true_labels_[labels>0], labels[labels>0])
		return score

	def _FE_error(self, estimated_FE_landscape):
		error = np.mean(np.abs(estimated_FE_landscape-self.true_FE_.FE_landscape_))
		return error

	def _density_error(self, estimated_density):
		error = np.mean(np.abs(estimated_density - self.true_density_))
		return error

	def visualize(self):
		"""
		Visualizing the quantities from estimations.
		:return:
		"""
		plt.figure(1)
		ax1 = plt.add_suplot(1,2,1)
		# Plot free energy error
		ax1.plot(self.FE_errors_GMM_CV_, linewidth=4, label='GMM with cross-validation')
		ax1.plot(self.FE_errors_GMM_mix_models_, linewidth=4, label='GMM with mixture of models')
		plt.legend()

		# Plot density error

		# Plot log-likelihood of test set

		# Plot clustering score

		plt.show()

		return
