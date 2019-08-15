import numpy as np
import free_energy_clustering.GMM as GMM
import scipy.optimize as opt

class LandscapeStacker(object):
    def __init__(self, data, list_of_validation_data, list_of_models, n_splits=1, convergence_tol=5e-3, n_iterations=1,
                 model_weights=None):
        """
        Class for weighting density estimators with EM, based on how well they describe the validation dataset.
        :param data: [n_samples x n_dimensions]
        :param list_of_estimators:
        :param n_splits: Number of folds in K-fold cross-validation
        :param convergence_tol:
        """
        self.GMM_list_ = list_of_models
        self.val_data_list_ = list_of_validation_data
        self.data_ = data
        self.convergence_tol_ = convergence_tol
        self.n_models_ = int(len(list_of_models)/n_splits)
        self.n_splits_ = n_splits
        self.n_iterations_ = n_iterations
        self.n_components_list_ = []

        # Initlialize weights
        if model_weights is None:
            if self.n_models_ > 0:
                self.model_weights_ = 1.0 / self.n_models_ * np.ones(self.n_models_)
        else:
            self.model_weights_ = model_weights
            self._sparisify_model()
            print('Model weights: ' + str(self.model_weights_))
            print('GMM list: '+str(self.GMM_list_))
            
        self._set_n_component_list()
        print('# Components in models: '+str(self.n_components_list_))
        return

    def objective_function(self,W):
        # -log(likelihood)
        W /= W.sum()
        return -self.loglikelihood(self.val_data_list_, list_of_validation_data=True, weights=W)

    def fit(self):
        do_EM = True

        print('Training density model weights.')

        if do_EM:
            loglikelihood = -np.inf
            prev_loglikelihood = 0
            while (np.abs(prev_loglikelihood - loglikelihood) > self.convergence_tol_):
                beta = self._expectation()
                self._maximization(beta)
                prev_loglikelihood = loglikelihood
                loglikelihood = self.loglikelihood(self.val_data_list_, list_of_validation_data=True)
        else:
            self.model_weights_ = opt.fmin_cg(self.objective_function, self.model_weights_)

        # Keep only models with nonzero weight
        self._sparisify_model()
        self._set_n_component_list()

        # Train each density model on the full dataset.
        print('Training each model on the full dataset.')
        for i_model in range(self.n_models_):
            n_components = self.GMM_list_[i_model].n_components_
            print(' - Training model with '+str(n_components)+' components')
            best_loglikelihood = -np.inf
            for i_iter in range(self.n_iterations_):
                density_model = GMM.GaussianMixture(n_components=n_components,
                                                    convergence_tol=self.convergence_tol_)
                density_model.fit(self.data_)
                loglikelihood = density_model.loglikelihood(self.data_)
                if loglikelihood > best_loglikelihood:
                    best_loglikelihood = loglikelihood
                    self.GMM_list_[i_model] = density_model

        self.n_components_list_ = np.asarray(self.n_components_list_)
        return

    def _set_n_component_list(self):
        """
        Set the list with number of components.
        :return:
        """
        self.n_components_list_ = []
        for i_model in range(self.n_models_):
            n_components = self.GMM_list_[i_model*self.n_splits_].weights_.shape[0]
            self.n_components_list_.append(n_components)
        return

    def _expectation(self):
        n_points = self.val_data_list_[0].shape[0]

        beta = np.zeros((self.n_splits_, self.n_models_, n_points))

        for i_split in range(self.n_splits_):
            for i_model in range(self.n_models_):
                ind = i_model*self.n_splits_+i_split
                beta[i_split, i_model, :] = self.model_weights_[i_model]*self.GMM_list_[ind].density(self.val_data_list_[ind])

            beta[i_split] /= np.sum(beta[i_split],axis=0)

        return beta

    def _maximization(self, beta):
        """
        Update density estimator weights.
        """
        self.model_weights_ = beta.sum(axis=(0,2))

        # Normalize Cat-distibution
        self.model_weights_ /= self.model_weights_.sum()
        return

    def _sparisify_model(self):
        """
        Remove all models with zero-weights (done after converged optimization).
        :return:
        """
        print('Removing zero-weighted models.')
        threshold = 1e-3
        n_models = np.sum(self.model_weights_>threshold)
        new_weights = []
        new_models = []

        for i_model in range(self.n_models_):
            if self.model_weights_[i_model] > threshold:
                new_weights.append(self.model_weights_[i_model])
                for i_split in range(self.n_splits_):
                    new_models.append(self.GMM_list_[i_model*self.n_splits_+i_split])

        self.n_models_ = n_models
        self.GMM_list_ = new_models
        print(self.GMM_list_)
        self.model_weights_ = np.asarray(new_weights)
        self.model_weights_ /= self.model_weights_.sum()
        return

    def density(self, x, list_of_validation_data=False, weights=None):
        """
        Compute mixture of landscape density at the given points, x.
        x is either a numpy-array of size [n_samples x n_dims] or a list of
        validation datasets with length [self.n_models_].
        """
        if list_of_validation_data:
            n_points = x[0].shape[0]
            density = np.zeros(n_points*self.n_splits_)
            for i_model in range(self.n_models_):
                for i_split in range(self.n_splits_):
                    if weights is None:
                        density[n_points*i_split:n_points*(i_split+1)] += self.model_weights_[i_model]*self.GMM_list_[i_model*self.n_splits_+i_split].density(x[i_model*self.n_splits_+i_split])
                    else:
                        density[n_points*i_split:n_points*(i_split+1)] += weights[i_model]*self.GMM_list_[i_model*self.n_splits_+i_split].density(x[i_model*self.n_splits_+i_split])
        else:
            density = np.zeros(x.shape[0])
            for i_model in range(self.n_models_):
                density += self.model_weights_[i_model]*self.GMM_list_[i_model].density(x)
        return density

    def loglikelihood(self, x, list_of_validation_data=False,weights=None):
        """
        Compute log-likelihood.
        """
        density = self.density(x, list_of_validation_data=list_of_validation_data,weights=weights)
        density[density<1e-8]=1e-8
        return np.mean(np.log(density))
