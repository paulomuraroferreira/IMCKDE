import utils
import numpy as np
import time
from sklearn.metrics import silhouette_score, precision_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import minimize

def improvedmulticlusterkde(adam, data, alpha, beta, n_clusters):
    """
    Improved multi-cluster KDE algorithm.

    Parameters:
    adam (bool): Whether to use ADAM optimizer.
    data (array-like): Data points.
    n_clusters (int): Number of clusters. Optional.
    alpha (float): Smoothing parameter.
    beta (float): Threshold multiplier.

    Returns:
    tuple: Cluster labels, centroids, and timing information.
    """
    centroids = []
    remaining_data = data.copy()
    i = 0
    time_taken_list = []
    m, n = data.shape

    while (i < n_clusters and len(remaining_data) != 0):    
        initial_point = remaining_data[0]
        m, n = remaining_data.shape  
        h_matrix = utils.h(remaining_data, alpha, m)
        hinverse = utils.h_inverse(h_matrix)
        gradient_function = lambda z: utils.gradient(z, remaining_data, h_1=h_matrix, h_inv=hinverse, m=m, n=n)
        function_to_minimize = lambda y: utils.f(y, data=remaining_data, h_1=h_matrix, h_inv=hinverse, m=m, n=n) 
        dp = utils.sigma(remaining_data, alpha, m)

        if adam:
            start_time = time.time()
            x_maximum = utils.adam_optimizer(grad_func=gradient_function, x_init=initial_point)
            end_time = time.time()
        else:
            start_time = time.time()
            x_maximum = minimize(function_to_minimize, initial_point, method='BFGS', jac=gradient_function).x
            end_time = time.time()
        time_taken_list.append((end_time-start_time))
        x_maximum = np.round(x_maximum, 2)
        remaining_data = np.delete(remaining_data, 0, axis=0)

        if not any(np.array_equal(x_maximum, centroid) for centroid in centroids):          
            i += 1
            threshold = dp * beta
            dists = np.linalg.norm(remaining_data - x_maximum, axis=1)
            mask = dists >= threshold
            remaining_data = remaining_data[mask]
            centroids.append(x_maximum)  
    
    clusters_original = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1) 
    return clusters_original, centroids, time_taken_list


class IMCKDE:
    def __init__(self, dataset, alpha=2, beta=1, n_clusters=None, adam=False):
        """
        Initialize the IMCKDE class.

        Parameters:
        dataset (array-like): Data points.
        n_clusters (int): Number of clusters.
        alpha (float): Smoothing parameter.
        beta (float): Threshold multiplier.
        adam (bool): Whether to use ADAM optimizer.
        """
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta
        # If no number of cluster is provided, it is set to inf.
        self.n_clusters = n_clusters if n_clusters is not None else np.inf
        self.result = None
        self.output_array_attr = None
        self.adam = adam
        self.time = 0

    @property
    def output_array(self):
        if self.output_array_attr is None:
            error_message = '''In order to get the attribute output_array it is necessary to execute the metrics_calculation function first.'''
            raise KeyError(str(error_message))
        
        else:
            return self.output_array_attr

    def predict(self):
        """
        Run the improved multi-cluster KDE algorithm.

        Returns:
        self: Fitted IMCKDE object.
        """
        results = improvedmulticlusterkde(adam=self.adam, data=self.dataset, n_clusters=self.n_clusters, alpha=self.alpha, beta=self.beta)
        self.result = results
        return self

    def centroids(self):
        """
        Get the centroids of the clusters.

        Returns:
        array-like: Centroids of the clusters.
        """
        return self.result[1]

    def clusters(self):
        """
        Get the cluster labels for the data points.

        Returns:
        array-like: Cluster labels.
        """
        return self.result[0]
    
    def metrics_calculation(self, metric, target=None):
        """
        Calculate the specified metric for the clustering.

        Parameters:
        metric (str): Metric to calculate ('silhouette', 'precision', 'db', 'dunn', 'ch').
        target (array-like): True labels (for precision and mapping metrics).

        Returns:
        float: Metric value.
        """

        if metric == 'silhouette':
            return silhouette_score(self.dataset, self.result[0])
       
        if metric in ['precision', 'db', 'dunn', 'ch']:
            from assignment_problem import ClusterMapper
            mapper = ClusterMapper(self.n_clusters)
            self.output_array_attr = mapper.mapping__clusters(target, self.dataset, self.result)
        
        if metric == 'precision':
            return precision_score(target, self.output_array, average='weighted')
        
        if metric == 'db':
            return davies_bouldin_score(self.dataset, self.output_array)
        
        if metric == 'dunn':
            return utils.dunn_index(self.output_array, self.dataset)
        
        if metric == 'ch':
            return calinski_harabasz_score(self.dataset, self.output_array)
