import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def multiclusterkdealgorithm(X, alpha, nc=0, max_iterations=1000, tolerance=1e-2):
    n, m = X.shape  # n - number of observations, m - dimension of data
    dp = np.std(X, axis=0)  # standard deviation of data

    # Matrix H (stored as vector)
    H = 1.06 * dp * m ** (-1 / alpha)

    # Kernel function with negative sign
    def kernel(x):
        aux1 = 1 / (m * np.sqrt((2 * np.pi) ** m) * np.sqrt(np.prod(H)))
        aux2 = np.sum(np.exp(-0.5 * np.sum(((x - X) / H) ** 2, axis=1)))
        return -aux1 * aux2  # -f

    # Not sure if the calculation of the gradient is correct
    def gradiente_kernel(x):
        aux1 = 1 / (m * np.sqrt((2 * np.pi) ** m) * np.sqrt(np.prod(H)))
        aux2 = np.exp(-0.5 * np.sum(((x - X) / H) ** 2, axis=1))
        aux3 = -(1 / H) * (x - X)
        return -aux1 * np.sum(aux2[:, None] * aux3, axis=0)  # -gradient

    # Distance from a point to a set of data
    def f_distancia(x, X):
        return np.sqrt(np.sum((x - X) ** 2, axis=1))

    # Discovery of cluster centers
    S = None
    flag = True
    pertence = False
    Distancias = None
    k = 0

    # Initial point for the optimizer
    inicio = X[0, :]

    while flag and k < max_iterations:
        sol = minimize(kernel, inicio, method="BFGS", jac=gradiente_kernel, options={"maxiter": 100, "gtol": 1e-4})
        otimo = np.round(sol.x, 3)  # stationary point

        if S is not None:
            pertence = np.any(np.all(np.isclose(S, otimo, atol=tolerance), axis=1))  # Increase the tolerance

        if pertence:
            flag = False
        else:
            S = np.atleast_2d(otimo) if S is None else np.vstack([S, otimo])
            dis = f_distancia(otimo, X)
            Distancias = dis if Distancias is None else np.column_stack([Distancias, dis])

            if k == 0:
                inicio = X[np.argmax(dis), :]
            else:
                a = np.min(Distancias, axis=1)
                inicio = X[np.argmax(a), :]

        k += 1

    # Number of clusters
    k -= 1

    if nc > 0 and nc < k:
        S = S[:nc, :]
        Distancias = Distancias[:, :nc]

    # Assigning points to centroids
    clusters = np.argmin(Distancias, axis=1)

    return {"centros": S, "cluster": clusters, "Distancias": Distancias, "otimo": otimo, "sol": sol}
