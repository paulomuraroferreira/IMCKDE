import numpy as np
from scipy.optimize import linear_sum_assignment

class ClusterMapper:
    def __init__(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters

    def mapping_(self, centroide_rotulado, centroide_calculado):
        # Calculate the cost matrix
        cost_matrix = np.zeros((self.number_of_clusters, self.number_of_clusters))
        for i in range(self.number_of_clusters):
            for j in range(self.number_of_clusters):
                cost_matrix[i, j] = np.linalg.norm(centroide_calculado[i] - centroide_rotulado[j])

        # Use the linear_sum_assignment function to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return row_ind, col_ind

    @staticmethod
    def map_values(input_array, mapping):
        # Generate temporary unique values
        temp_values = {i: max(input_array.max(), max(mapping.values())) + 1 + i for i in range(len(mapping))}

        # First, replace with temporary unique values
        for row_value in mapping.keys():
            input_array = np.where(input_array == row_value, temp_values[row_value], input_array)
            
        # Then replace temporary unique values with the final ones
        for row_value, col_value in mapping.items():
            input_array = np.where(input_array == temp_values[row_value], col_value, input_array)
        
        return input_array

    def mapping__clusters(self, target, dataset, result__):
        n_clusters = len(np.unique(target))
        centroid_geo = {}
        for i in range(n_clusters):
            centroid_geo[i] = np.mean(dataset[np.where(target == i)], axis=0)

        centroides_multicluster = result__[1]
        row_ind, col_ind = self.mapping_(np.array([value for key, value in centroid_geo.items()]), centroides_multicluster)
        mapping = {row_ind[i]: col_ind[i] for i in range(n_clusters)}
        input_array = np.array(result__[0])
        return self.map_values(input_array, mapping)

    

if __name__ == "__main__":
    # Example usage with the Iris dataset
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    iris = load_iris()
    target = iris.target
    n_clusters = 3
    dataset = iris.data
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(iris.data)
    result = (kmeans.labels_, kmeans.cluster_centers_,)
    print(result)
    mapper = ClusterMapper(n_clusters)
    maps = mapper.mapping__clusters(target, dataset, result)
    print(maps)

    from sklearn.metrics import precision_score
    print(precision_score(target, maps, average='weighted'))
