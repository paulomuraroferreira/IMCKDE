import math
import time
import numpy as np
import pandas as pd
import statistics
import scipy
from scipy.optimize import minimize_scalar
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.optimize import fsolve
from sklearn.metrics import silhouette_score
import random
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from numba import jit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import precision_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
import multiprocessing
from sklearn.metrics.pairwise import euclidean_distances
import numba
import pickle
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import calinski_harabasz_score




def dunn_index(labels, data):
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
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**(max_iter + 1))
    v_hat = v / (1 - beta2**(max_iter + 1))
    x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x, m, v

def adam_optimizer(grad_func, x_init, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
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


# m é o número de pontos; n é a dimensão;
def distancia(dado, centro):
  dists = (np.array(dado) - np.array(centro))**2
  return np.sqrt(dists.sum())

def h(dados, alpha, m):
  #m = dados.shape[0]
  return 1.06*np.diag(np.std(np.transpose(dados), axis = 1))*m**(-1/alpha)

def h_inversa(h_matrix):
  return np.diag(1/(h_matrix.diagonal()))

def f(x, dados, h_1, h_inv, m, n):
  #h_1 = h(dados, alpha, m)
  #m = dados.shape[0]
  #n = dados.shape[1]
  return -((2*math.pi)**(-n/2))*(1/m)*(np.linalg.det(h_1))**(-1/2)* \
        sum([math.exp(np.matmul((-1/2) * np.transpose((x - dados[i])),np.matmul(h_inv,(x - dados[i])))) for i in range(m)])

def gradient(x, dados, h_1, h_inv, m, n):
  #h_1 = h(dados, alpha, m)
  #m = dados.shape[0]
  #n = dados.shape[1]
  return -((2*math.pi)**(-n/2))*(1/m)*(np.linalg.det(h_1))**(-1/2)* \
        sum([(-1)*np.matmul(h_inv, (x - dados[i]))*math.exp(np.matmul((-1/2) \
            * np.transpose((x - dados[i])),np.matmul(h_inv,(x - dados[i])))) for i in range(m)])


def sigma(data, alpha, m):
    return np.sqrt(1.06 * m**(-1/alpha) * np.sum(np.std(data, axis=0)**2))



def multicluster_single(adam, dados, n_clusters, alpha, multiplo):
    centroides = []
    dados_restantes = dados.copy()
    i = 0
    lista_aux = []
    m, n = dados.shape

    while (i < n_clusters and len(dados_restantes) != 0):    
        initial_point = dados_restantes[0]
        m, n = dados_restantes.shape  
        h_matrix = h(dados_restantes, alpha, m)
        hinversa = h_inversa(h_matrix)
        gradiente_fun2 = lambda z: gradient(z, dados_restantes, h_1=h_matrix, h_inv = hinversa, m=m, n=n)
        funcao = lambda y: f(y, dados = dados_restantes, h_1 = h_matrix, h_inv = hinversa, m=m, n=n) 
        dp = sigma(dados_restantes, alpha, m)

        if adam:
            start_time = time.time()
            x_maximo = adam_optimizer(grad_func = gradiente_fun2, x_init = initial_point)
            end_time = time.time()
            
        else:
            start_time = time.time()
            x_maximo = minimize(funcao, initial_point, method='BFGS', jac=gradiente_fun2).x
            end_time = time.time()
        lista_aux.append((end_time-start_time))
        x_maximo = np.round(x_maximo, 2)
        dados_restantes = np.delete(dados_restantes, 0, axis = 0)

        if not any(np.array_equal(x_maximo, centroide) for centroide in centroides):          
            i += 1
            threshold = dp * multiplo

            dists = np.linalg.norm(dados_restantes - x_maximo, axis = 1)
            mask = dists >= threshold

            dados_restantes = dados_restantes[mask]
            centroides.append(x_maximo)  
            
    clusters_original = np.argmin(np.linalg.norm(dados[:, np.newaxis] - centroides, axis=2), axis=1) 
    return clusters_original , centroides, lista_aux

def funcao(args):
    ponto_inicial, dados_restantes, funcao_kde, gradiente_fun2 = args
    
    initial_point = ponto_inicial
    minimizacao = minimize(funcao_kde, initial_point, method='BFGS', jac=gradiente_fun2)
    x_maximo = np.round(minimizacao.x,2)
    f_maxima = minimizacao.fun
    
    return x_maximo, f_maxima

def multicluster_parallel(adam, dados, n_clusters, alpha, multiplo, initializer):
    np.random.seed(0) 
    centroides = []
    f_maximas = []
    dados_restantes = dados.copy()
    contador = 0   
    #m, n = dados.shape
    lista_aux = []
    while (contador < n_clusters and len(dados_restantes) != 0):  
        if initializer == 'kmeans':
            points = KMeans(n_clusters=(n_clusters-len(centroides)), random_state = 0, n_init = 'auto').fit(dados_restantes).cluster_centers_
        elif initializer == 'random':
            points = dados_restantes[np.random.choice(dados_restantes.shape[0], size=(n_clusters-len(centroides)), replace=False), :]
        else:
            raise ValueError("Invalid initializer. Expected 'kmeans' or 'random'.")

        m, n = dados_restantes.shape  
        h_matrix = h(dados_restantes, alpha, m)
        hinversa = h_inversa(h_matrix)
        
        gradiente_fun2 = partial(gradient, dados=dados_restantes, h_1=h_matrix, h_inv=hinversa, m=m, n=n)
        funcao_kde = partial(f, dados = dados_restantes, h_1 = h_matrix, h_inv = hinversa, m=m, n=n)
        
        com = time.time()

        with Pool(processes=n_clusters) as pool:
            lista_x_maximos = pool.map(funcao, [(p, dados_restantes, funcao_kde, gradiente_fun2) for p in points])

        fin = time.time()
        lista_aux.append((fin-com))
        
        sub_arrays = points
        for sub_array in sub_arrays:
            indices = np.where(np.all(dados_restantes == sub_array, axis=1))
            dados_restantes = np.delete(dados_restantes, indices, axis=0)
        
        for elemento in lista_x_maximos:
            x_maximo = elemento[0]
            f_maxima = elemento[1]

            if not any(np.array_equal(x_maximo, centroide) for centroide in centroides):      
                contador += 1              
                dp = sigma(dados_restantes, alpha, m)
                threshold = dp * multiplo
                dists = np.linalg.norm(dados_restantes - x_maximo, axis = 1)
                mask = dists >= threshold
                dados_restantes = dados_restantes[mask]
                centroides.append(x_maximo)
                f_maximas.append(f_maxima)

    
    f_maximas,centroides = map(list, zip(*sorted(zip(f_maximas, centroides), key = lambda x: x[0])))
    centroides = centroides[-n_clusters:]
    f_maximas = f_maximas[-n_clusters:]
                                   
    clusters_original = np.argmin(np.linalg.norm(dados[:, np.newaxis] - centroides, axis=2), axis=1) 
    return clusters_original , centroides, lista_aux



class IMCKDE:
    def __init__(self, dataset, n_clusters, alpha=2, multiplo=1, paralelization = True, adam = False, initializer="kmeans"):
        self.dataset = dataset
        #self.target = target
        self.alpha = alpha
        self.multiplo = multiplo
        self.initializer = initializer
        self.n_clusters = n_clusters
        self.result = None
        self.output_array = None
        self.paralelization = paralelization
        self.adam = adam
        self.time = 0

    def predict(self):
        if self.paralelization:
            resultado = multicluster_parallel(adam = self.adam, dados = self.dataset, n_clusters = self.n_clusters, alpha = self.alpha, multiplo = self.multiplo, initializer = self.initializer)
        else:
            resultado = multicluster_single(adam = self.adam, dados = self.dataset, n_clusters=self.n_clusters, alpha = self.alpha, multiplo=self.multiplo)
        
        self.result = resultado

        return self

    def centroids(self):
        return self.result[1]

    def clusters(self):
        return self.result[0]
    
    def calcular_metricas(self, metrica, target = None):

        from assignment_problem import ClusterMapper

        if metrica == 'silhueta':
            return silhouette_score(self.dataset, self.result[0])
       
        if metrica in ['precision', 'db', 'dunn', 'ch']:


            mapper = ClusterMapper(self.n_clusters)
            self.output_array = mapper.mapeamento_clusters(target, self.dataset, self.result)

            # self.output_array = mapeamento_clusters(target, self.n_clusters, self.dataset, self.result)
            # print('self.output_array',self.output_array)
        
        if metrica == 'precision':
            return precision_score(target, self.output_array, average='weighted')
        
        if metrica == 'db':
            return davies_bouldin_score(self.dataset, self.output_array)
        
        if metrica == 'dunn':
            return dunn_index(self.output_array, self.dataset)
        
        if metrica == 'ch':
            return calinski_harabasz_score(self.dataset, self.output_array)
        

