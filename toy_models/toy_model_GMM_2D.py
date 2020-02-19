import numpy as np
from free_energy_clustering.GMM import GaussianMixture
from free_energy_clustering.GMM_free_energy import FreeEnergyClustering

class GMM2D(GaussianMixture):

    def __init__(self):
        GaussianMixture.__init__(self, n_components=9)

        self.n_dims_ = 2
        self._set_parameters()
        self.name = 'GMM_2D'
        return

    def _set_cov(self, x11,x12,x22):
        tmp_cov = np.zeros((self.n_dims_, self.n_dims_))

        tmp_cov[0, 0] = x11
        tmp_cov[0, 1] = x12
        tmp_cov[1, 0] = x12
        tmp_cov[1, 1] = x22
        return tmp_cov

    def _set_parameters(self):

        self.means_ = np.asarray([ np.asarray([0.8,0.35]), np.asarray([0.45,0.52]), np.asarray([0.2,0.6]),
                                   np.asarray([0.05,0.8]), np.asarray([0.5,0.25]), np.asarray([0.5,0.25]),
                                   np.asarray([0.5, 0.25]), np.asarray([0.4, 0.34]), np.asarray([0.8,0.5])])

        covs = [np.zeros((2,2))]*self.n_components_

        covs[0] = self._set_cov(0.0021, 0.0005, 0.002)
        covs[1] = self._set_cov(0.001, 0.0009, 0.001)
        covs[2] = self._set_cov(0.002, 0.001, 0.002)
        covs[3] = self._set_cov(0.003, 0.0008, 0.003)
        covs[4] = self._set_cov(0.0012, 0.0005, 0.0012)
        covs[5] = self._set_cov(0.005, 0.0, 0.0015)
        covs[6] = self._set_cov(0.002, 0.0, 0.002)
        covs[7] = self._set_cov(0.0012, 0.00, 0.002)
        covs[8] = self._set_cov(0.001, 0.0009, 0.001)

        self.covariances_ = covs

        self.weights_ = np.asarray([0.15,0.1,0.3,0.25,0.1,0.05,0.1,0.05,0.4])
        self.weights_ /= np.sum(self.weights_)

        return
