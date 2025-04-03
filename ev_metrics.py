import numpy as np

class Utils:
    @staticmethod
    def get_centroids(M, features):
        num_samples = M.shape[0]
        num_metaclusters = M.shape[1]
        feat_dim = features.shape[1]
        
        centroids = np.zeros((num_metaclusters, feat_dim))
        for i in range(num_metaclusters):
            for j in range(num_samples):
                centroids[i] += M[j, i] * features[j] / np.sum(M[:, i])

        return centroids[1:]  # remove the first centroid, which is the outliers

class EvidentialMetrics:
    def __init__(self, F, features, distance_fn=lambda x, y: np.linalg.norm(x - y)):
        self.F = F
        self.features = features

        self.number_observations = features.shape[0]
        self.number_metaclusters = F.shape[0]
        self.number_clusters = F.shape[1]
        self.distance_fn = distance_fn

        # Compute the distance matrix
        distance_matrix = np.zeros((self.number_observations, self.number_observations))
        for i in range(self.number_observations):
            for j in range(i + 1, self.number_observations):
                distance_matrix[i, j] = self.distance_fn(self.features[i], self.features[j])
                distance_matrix[j, i] = distance_matrix[i, j]
        self.distance_matrix = distance_matrix


    def get_sillhouette(self, assignability_fn, outlier_coefficient=10, normalized=True):
        def sillhouette_fn(mass):
            # function to compute the average distance between points and metaclusters
            def a(assignabilit_matrix):
                out = np.dot(self.distance_matrix, assignabilit_matrix)
                out /= assignabilit_matrix.sum(axis=0)
                return out
            
            assignability_mat = assignability_fn(mass)
            # Compute the a, average distance to clusters
            a = a(assignability_mat)

            # compute b
            b = np.ones(a.shape)*np.inf
            for metacluster in range(self.number_metaclusters):
                for point in range(self.number_observations):
                    for B in range(self.number_metaclusters):
                        if not B == metacluster:
                            if a[point, B] < b[point, metacluster]:
                                b[point, metacluster] = a[point, B]
            # compute sillhouette
            s = (b-a)/np.maximum(a, b)
            if normalized:            
                sillhouette = (assignability_mat*s).sum(axis=1)/assignability_mat.sum(axis=1)
            else:
                sillhouette = (assignability_mat*s).sum(axis=1)
                
            return sillhouette.mean()

        return sillhouette_fn
        