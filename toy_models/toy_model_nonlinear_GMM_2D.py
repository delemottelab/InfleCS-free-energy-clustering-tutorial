import numpy as np
from free_energy_clustering.GMM import GaussianMixture
from free_energy_clustering.GMM_free_energy import FreeEnergyClustering

class GMM2dNonlinear():

    def __init__(self, transform_data=True):

        self.transform_data=transform_data
        self.n_dims_ = 2
        self.name = 'nonlinear_GMM_2D'
        n_components = 3
        self.GMM = GaussianMixture(n_components=n_components)
        self.n_components_ = n_components
        self._set_parameters()
        return

    def transform(self,x):
        x = np.asarray([x[:,0], x[:,1]+(2.0*x[:,0]**3)]).T # np.sqrt(np.abs(x[:,0]))]).T
        return x

    def detransform(self,x):
        x = np.asarray([x[:,0], x[:,1]-(2.0*x[:,0]**3)]).T # np.sqrt(np.abs(x[:,0]))]).T
        return x

    def sample(self, n_points):
        x = self.GMM.sample(n_points)
        if self.transform_data:
            x = self.transform(x)
        return x

    def assign_cluster_labels(self,x):
        if self.transform_data:
            x = self.detransform(x)
        labels = self.GMM.predict(x)+1
        #labels[labels==3] = 2
        return labels

    def density(self, x):
        if self.transform_data:
            x = self.detransform(x)
        return self.GMM.density(x)

    def _set_cov(self, x11,x12,x22):
        tmp_cov = np.zeros((self.n_dims_, self.n_dims_))

        tmp_cov[0, 0] = x11
        tmp_cov[0, 1] = x12
        tmp_cov[1, 0] = x12
        tmp_cov[1, 1] = x22
        return tmp_cov

    def _set_parameters(self):

        #self.GMM.means_ = np.asarray([np.asarray([0.0,0.6]), np.asarray([0.3,0.25]), np.asarray([0.3,0.25])])
        self.GMM.means_ = np.asarray([np.asarray([-0.8, 0.6]), np.asarray([-0.5, 0.25]), np.asarray([-0.6, 0.25])])
        covs = [np.zeros((2,2))]* self.GMM.n_components_

        covs[0] = self._set_cov(0.01, 0.005, 0.05)
        covs[1] = self._set_cov(0.05, -0.01, 0.015)
        covs[2] = self._set_cov(0.001, 0.000, 0.01)
        #covs[1] = self._set_cov(0.05, -0.01, 0.015)
        #covs[2] = self._set_cov(0.001, 0.000, 0.01)

        self.GMM.covariances_ = covs

        self.GMM.weights_ = np.asarray([0.8,0.08,0.08])
        self.GMM.weights_ /= np.sum(self.GMM.weights_)

        return
