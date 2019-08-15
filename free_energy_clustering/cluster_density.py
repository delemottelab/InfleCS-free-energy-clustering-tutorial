import numpy as np
from scipy.spatial.distance import cdist


class ClusterDensity(object):

	def __init__(self, points, eval_points=None):
		self.grid_points_ = points
		self.points_ = eval_points
		self.grid_cluster_inds_ = None
		return
	
	def _construct_components(self,distance_matrix, is_FE_min, use_transition_matrix=False):
		# Build subgraphs with connected components of the isolated FE minima
		print('Constructing connected components.')
		n_points = distance_matrix.shape[0]
		
		graph = np.zeros((n_points,n_points))
		
		if use_transition_matrix:
			# Set distance matrix to information distance of transition matrix: 
			# Kinetically close states have high transition probability but should have low distance
			distance_matrix = -np.log(distance_matrix+1e-9)
		
		# Sort distances in ascending order
		sort_inds = np.argsort(distance_matrix,axis=1)
		
		for i in range(n_points):
			if is_FE_min[i]:
				check_points = []
				neighbors = sort_inds[i,:]
				k_neighbors=1
				
				# Add neighbors until another potential component is reached
				for j in range(k_neighbors,n_points):
					current_neighbor = neighbors[j]
					if is_FE_min[current_neighbor]:
				
						neighbor_distance = distance_matrix[i,current_neighbor]
				
						if len(check_points) > 2:
							check_point_distances = distance_matrix[current_neighbor,np.asarray(check_points)]
							is_smaller_dist = check_point_distances < neighbor_distance
							if np.sum(is_smaller_dist) > 0:
								# A non-component point is closer to both the current point and 
								# the other component point => the two component points are not neighbors
								break;
				
						# Add connection between neighbors
						graph[i,current_neighbor] = 1	
						# Enforce symmetry
						graph[current_neighbor,i] = 1			
					else:
						check_points.append(current_neighbor)
	
		# Sparsify graph to contain only the connected components
		graph = graph[is_FE_min,:]
		graph = graph[:,is_FE_min]

		return graph

	def _find_connected_components(self,graph):
		# Assign points to connected components
		print('Clustering data points.')
		
		n_points = graph.shape[0]
		component_indices = np.zeros(n_points)
		is_visited = np.zeros(n_points)	
		all_inds = np.arange(n_points)
	
		i_component = 0
		while np.sum(is_visited) < is_visited.shape[0]:
			i_component += 1
			queue = []
			# get next unvisited point
			unvisited_points = all_inds[is_visited==0]
			queue.append(unvisited_points[0])
		
			while len(queue) > 0:
				current_point = queue.pop(0)
				if is_visited[current_point] == 0:
					is_visited[current_point] = 1
					component_indices[current_point] = i_component
	
					# get unvisited neighbors 
					neighbors = all_inds[graph[current_point,:] > 0]
					for neighbor in neighbors:
						if is_visited[neighbor] == 0:
							queue.append(neighbor)

		return component_indices
	
	def data_cluster_indices(self, point_distances, cluster_indices_eval_points):
		"""
		Set cluster indices according to the closest data point.
		"""
		n_points = point_distances.shape[0]
		cluster_inds = np.zeros(n_points)
		
		min_inds = np.argmin(point_distances,axis=1)
		
		# Set cluster index of point to the same as the cluster index of evaluated (grid) point
		cluster_inds = cluster_indices_eval_points[min_inds]
		return cluster_inds

	def cluster_data(self, is_FE_min, transition_matrix=None):
		
		# Construct and detect connected components
		if transition_matrix is None:
			graph = self._construct_components(cdist(self.grid_points_,self.grid_points_), is_FE_min)
		else:
			print('Using transition probabilities to define distances.')
			graph = self._construct_components(transition_matrix, is_FE_min, use_transition_matrix=True)
		
		print('# Graph connections: '+str(np.sum(graph)))
		cluster_indices_grid_points = self._find_connected_components(graph)
		
		self.grid_cluster_inds_ = np.zeros(self.grid_points_.shape[0])
		self.grid_cluster_inds_[is_FE_min] = cluster_indices_grid_points
		if self.points_ is not None:
			cluster_indices = self.data_cluster_indices(cdist(self.points_,self.grid_points_),self.grid_cluster_inds_)
		else:
			cluster_indices = self.grid_cluster_inds_
		
		return cluster_indices
