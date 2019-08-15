import numpy as np
import free_energy_clustering as FEC
import sys

class FreeEnergyPathways(FEC.LandscapeClustering):

    def __init__(self, density_model, data, temperature=300, n_points=100, convergence_tol=1e-1, step_size=1e-3,
                 ensemble_of_GMMs=False, max_iter=1000):

        FEC.LandscapeClustering.__init__(self,ensemble_of_GMMs=ensemble_of_GMMs)

        self.density_model_ = density_model
        self.convergence_tol_ = convergence_tol
        self.n_points_ = n_points
        self.data_ = data
        self.n_dims_ = data.shape[1]
        self.temperature_ = temperature # [K]
        self.boltzmann_constant_ = 0.0019872041 # [kcal/(mol K)]
        self.step_size_ = step_size
        self.max_iterations_ = max_iter
        return

    def _initialize_path(self, state_from, state_to):
        """
        Set initial path guess as straight path between the two states.
        :param state_from:
        :param state_to:
        :return:
        """
        path = np.zeros((self.n_points_, self.n_dims_))
        for i_dim in range(self.n_dims_):
            path[:,i_dim] = np.linspace(state_from[i_dim],state_to[i_dim],num=self.n_points_)
        return path

    def _length_of_subpaths(self, path):
        partial_path_lengths = np.zeros(self.n_points_)

        for i in range(1,self.n_points_):
            partial_path_lengths[i] = partial_path_lengths[i-1]+np.linalg.norm(path[i]-path[i-1])

        subpath_points = np.arange(self.n_points_)*partial_path_lengths[-1]/(self.n_points_-1)

        return partial_path_lengths, subpath_points

    def _equilibrate_path_points(self, path):
        """
        Spread points equidistantly along path.
        :param path:
        :return:
        """
        partial_path_lengths, subpath_points = self._length_of_subpaths(path)

        new_path = path
        for i in range(1,self.n_points_-1):
            s = subpath_points[i]
            for j in range(1,self.n_points_):
                if s > partial_path_lengths[j-1] and s < partial_path_lengths[j]:
                    new_path[i] = path[j-1] + (s-partial_path_lengths[j-1])*(path[j]-path[j-1])/np.linalg.norm(path[j]-path[j-1])
                    break

        return new_path

    def _update_path(self, path):
        """
        Update path with one minimization and equilibration of path points.
        :param path:
        :return:
        """
        density = self.density_model_.density(path)
        density[density<1e-15]=1e-15

        inner_derivative, _ = self._compute_gradients(self.density_model_, path)
        outer_derivative = -self.temperature_*self.boltzmann_constant_/density
        step = self.step_size_*(np.multiply(outer_derivative[:,np.newaxis],inner_derivative))
        new_path = path - step
        new_path = self._equilibrate_path_points(new_path)
        return new_path

    def minimum_pathway(self, state_from, state_to):
        """
        Compute minimum pathway between two states using the estimated free energy landscape based on GMM.
        :param state_from:
        :param state_to:
        :return:
        """

        # Set linear path between end points
        path = self._initialize_path(self.data_[state_from], self.data_[state_to])

        prev_path = np.inf * path
        counter = 1
        while np.linalg.norm(path-prev_path) > self.convergence_tol_:
            sys.stdout.write("\r" + 'Iteration: ' + str(counter) + '/' + str(self.max_iterations_))
            sys.stdout.flush()
            prev_path = np.copy(path)
            path = self._update_path(prev_path)
            if counter >= self.max_iterations_:
                break
            counter+=1
        print()
        return path

