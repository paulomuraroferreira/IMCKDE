from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.text
import math
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
import numba

def plot_confusion_matrix(y_true, y_pred, labels, filename, title):
    """
    Plots and saves a confusion matrix with the given true and predicted labels, label names, and a title.

    Parameters:
        y_true (array-like): Array of true labels.
        y_pred (array-like): Array of predicted labels by the model.
        labels (list of str): List of label names corresponding to the classes in the confusion matrix.
        filename (str): The filepath where the image will be saved.
        title (str): The title for the confusion matrix plot.

    This function generates a confusion matrix from the true and predicted labels using scikit-learn's `confusion_matrix`,
    and then visualizes it using matplotlib's `ConfusionMatrixDisplay`. The font size of the plot title and the matrix
    labels are set to enhance readability. After adjusting the font sizes, the plot is saved as a high-resolution JPEG
    image and also displayed.

    Returns:
        None. This function only creates and saves a plot.

    Notes:
        The function checks if the `ConfusionMatrixDisplay` object has a `text_` attribute for direct font size adjustment.
        If `text_` is not present, it tries to modify the font size of all text elements found among the axes children.
        It is assumed that `matplotlib.pyplot` and relevant functions from `sklearn.metrics` have been imported externally.
    """
    # Generate confusion matrix from true and predicted labels
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Display the matrix with a title
    disp.plot()
    plt.title(title, fontsize=20)

    # Modify text elements directly if available
    if hasattr(disp, 'text_') and disp.text_ is not None:
        for texts in disp.text_:
            for text in texts:
                text.set_fontsize(20)  # Set font size to 20
    else:
        # Fallback: Iterate over ax children to find and modify Text objects
        for child in disp.ax_.get_children():
            if isinstance(child, matplotlib.text.Text):
                child.set_fontsize(20)  # Set font size to 20

    # Save the figure
    plt.savefig(filename, dpi=300, format='jpeg')
    plt.show()

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


# def funcao(args):
#     """
#     Wrapper function for minimization.

#     Parameters:
#     args (tuple): Arguments for the minimization function.

#     Returns:
#     tuple: Optimized point and function value.
#     """
#     initial_point_, remaining_data, function_kde, gradient_fun2 = args
#     initial_point = initial_point_
#     minimization_ = minimize(function_kde, initial_point, method='BFGS', jac=gradient_fun2)
#     x_maximum = np.round(minimization_.x, 2)
#     f_maxima = minimization_.fun
#     return x_maximum, f_maxima