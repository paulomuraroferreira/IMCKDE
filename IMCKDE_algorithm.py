import math
import time
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score, precision_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances
import numba

def dunn_index(labels, data):
    """
    Compute the Dunn index for a given clustering.

    Parameters:
    labels (array-like): Cluster labels for each data point.
    data (array-like): Data points.

    Returns:
    float: Dunn index value.
    """
    distances = euclidean_distances(data, data)
    labels = np.array(labels)
    clusters = np.unique(labels)
    intra_cluster_distances = []
    inter_cluster_distances = []

    for cluster in clusters:
        data_in_cluster = data[labels == cluster]
        # Compute intra-cluster distances for this cluster and append to list
        intra_cluster_distances.append(distances[labels == cluster][:, labels == cluster].max())
        # Compute inter-cluster distances for this cluster compared to all other clusters
        for other_cluster in clusters:
            if cluster != other_cluster:
                inter_cluster_distances.append(distances[labels == cluster][:, labels == other_cluster].min())

    dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
    return dunn_index

@numba.jit(nopython=True)
def adam_optimizer_inner(grad, x, m, v, learning_rate, beta1, beta2, epsilon, max_iter):
    """
    Inner loop of the ADAM optimizer with Numba JIT for speed.

    Parameters:
    grad (array-like): Gradient.
    x (array-like): Current parameter values.
    m (array-like): First moment vector.
    v (array-like): Second moment vector.
    learning_rate (float): Learning rate.
    beta1 (float): Exponential decay rate for the first moment estimates.
    beta2 (float): Exponential decay rate for the second moment estimates.
    epsilon (float): Small value to avoid division by zero.
    max_iter (int): Maximum number of iterations.

    Returns:
    tuple: Updated (x, m, v).
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**(max_iter + 1))
    v_hat = v / (1 - beta2**(max_iter + 1))
    x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x, m, v

def adam_optimizer(grad_func, x_init, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
    """
    ADAM optimizer implementation.

    Parameters:
    grad_func (callable): Function to compute the gradient.
    x_init (array-like): Initial parameter values.
    learning_rate (float): Learning rate.
    beta1 (float): Exponential decay rate for the first moment estimates.
    beta2 (float): Exponential decay rate for the second moment estimates.
    epsilon (float): Small value to avoid division by zero.
    max_iter (int): Maximum number of iterations.

    Returns:
    array-like: Optimized parameters.
    """
    x = np.array(x_init, dtype=np.float64)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for i in range(max_iter):
        grad = np.array(grad_func(x), dtype=np.float64)
        x, m, v = adam_optimizer_inner(grad, x, m, v, learning_rate, beta1, beta2, epsilon, max_iter)
        if np.allclose(x, x_init, atol=1e-6, rtol=1e-6, equal_nan=False):
            break
        x_init = x
    return x

def distance(dado, centro):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    dado (array-like): First point.
    centro (array-like): Second point.

    Returns:
    float: Euclidean distance.
    """
    dists = (np.array(dado) - np.array(centro))**2
    return np.sqrt(dists.sum())

def h(data, alpha, m):
    """
    Compute the bandwidth matrix for KDE.

    Parameters:
    data (array-like): Data points.
    alpha (float): Smoothing parameter.
    m (int): Number of data points.

    Returns:
    array-like: Bandwidth matrix.
    """
    return 1.06 * np.diag(np.std(np.transpose(data), axis=1)) * m**(-1/alpha)

def h_inverse(h_matrix):
    """
    Compute the inverse of the bandwidth matrix.

    Parameters:
    h_matrix (array-like): Bandwidth matrix.

    Returns:
    array-like: Inverse bandwidth matrix.
    """
    return np.diag(1 / h_matrix.diagonal())

def f(x, data, h_1, h_inv, m, n):
    """
    KDE objective function.

    Parameters:
    x (array-like): Point at which to evaluate the KDE.
    data (array-like): Data points.
    h_1 (array-like): Bandwidth matrix.
    h_inv (array-like): Inverse bandwidth matrix.
    m (int): Number of data points.
    n (int): Dimensionality of data.

    Returns:
    float: KDE value at point x.
    """
    return -((2*math.pi)**(-n/2)) * (1/m) * (np.linalg.det(h_1))**(-1/2) * \
        sum([math.exp(np.matmul((-1/2) * np.transpose((x - data[i])), np.matmul(h_inv, (x - data[i])))) for i in range(m)])

def gradient(x, data, h_1, h_inv, m, n):
    """
    Gradient of the KDE objective function.

    Parameters:
    x (array-like): Point at which to evaluate the gradient.
    data (array-like): Data points.
    h_1 (array-like): Bandwidth matrix.
    h_inv (array-like): Inverse bandwidth matrix.
    m (int): Number of data points.
    n (int): Dimensionality of data.

    Returns:
    array-like: Gradient at point x.
    """
    return -((2*math.pi)**(-n/2)) * (1/m) * (np.linalg.det(h_1))**(-1/2) * \
        sum([(-1)*np.matmul(h_inv, (x - data[i])) * math.exp(np.matmul((-1/2) * np.transpose((x - data[i])), np.matmul(h_inv, (x - data[i])))) for i in range(m)])

def sigma(data, alpha, m):
    """
    Compute the standard deviation for KDE.

    Parameters:
    data (array-like): Data points.
    alpha (float): Smoothing parameter.
    m (int): Number of data points.

    Returns:
    float: Standard deviation.
    """
    return np.sqrt(1.06 * m**(-1/alpha) * np.sum(np.std(data, axis=0)**2))

def improvedmulticlusterkde(adam, data, n_clusters, alpha, beta):
    """
    Improved multi-cluster KDE algorithm.

    Parameters:
    adam (bool): Whether to use ADAM optimizer.
    data (array-like): Data points.
    n_clusters (int): Number of clusters.
    alpha (float): Smoothing parameter.
    beta (float): Threshold multiplier.

    Returns:
    tuple: Cluster labels, centroids, and timing information.
    """
    centroids = []
    remaining_data = data.copy()
    i = 0
    lista_aux = []
    m, n = data.shape

    while (i < n_clusters and len(remaining_data) != 0):    
        initial_point = remaining_data[0]
        m, n = remaining_data.shape  
        h_matrix = h(remaining_data, alpha, m)
        hinverse = h_inverse(h_matrix)
        gradient_fun2 = lambda z: gradient(z, remaining_data, h_1=h_matrix, h_inv=hinverse, m=m, n=n)
        funcao = lambda y: f(y, data=remaining_data, h_1=h_matrix, h_inv=hinverse, m=m, n=n) 
        dp = sigma(remaining_data, alpha, m)

        if adam:
            start_time = time.time()
            x_maximum = adam_optimizer(grad_func=gradient_fun2, x_init=initial_point)
            end_time = time.time()
        else:
            start_time = time.time()
            x_maximum = minimize(funcao, initial_point, method='BFGS', jac=gradient_fun2).x
            end_time = time.time()
        lista_aux.append((end_time-start_time))
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
    return clusters_original, centroids, lista_aux

def funcao(args):
    """
    Wrapper function for minimization.

    Parameters:
    args (tuple): Arguments for the minimization function.

    Returns:
    tuple: Optimized point and function value.
    """
    initial_point_, remaining_data, function_kde, gradient_fun2 = args
    initial_point = initial_point_
    minimization_ = minimize(function_kde, initial_point, method='BFGS', jac=gradient_fun2)
    x_maximum = np.round(minimization_.x, 2)
    f_maxima = minimization_.fun
    return x_maximum, f_maxima

class IMCKDE:
    def __init__(self, dataset, n_clusters, alpha=2, beta=1, adam=False):
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
        self.n_clusters = n_clusters
        self.result = None
        self.output_array = None
        self.adam = adam
        self.time = 0

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
        from assignment_problem import ClusterMapper

        if metric == 'silhouette':
            return silhouette_score(self.dataset, self.result[0])
       
        if metric in ['precision', 'db', 'dunn', 'ch']:
            mapper = ClusterMapper(self.n_clusters)
            self.output_array = mapper.mapping__clusters(target, self.dataset, self.result)
        
        if metric == 'precision':
            return precision_score(target, self.output_array, average='weighted')
        
        if metric == 'db':
            return davies_bouldin_score(self.dataset, self.output_array)
        
        if metric == 'dunn':
            return dunn_index(self.output_array, self.dataset)
        
        if metric == 'ch':
            return calinski_harabasz_score(self.dataset, self.output_array)
