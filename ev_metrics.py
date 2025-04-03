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
    
    @staticmethod
    def jaccard_matrix_calculus(number_metaclasses):
        """
        Compute the Jaccard matrix for the disernment framework of the given mass function
        """
        natoms = round(np.log2(number_metaclasses))
        ind = [{}]*number_metaclasses
        if (np.power(2, natoms) == number_metaclasses):
            ind[0] = {0} #In fact, the first element should be a None value (for empty set).
            #But in the following calculate, we'll deal with 0/0 which shoud be 1 bet in fact not calculable. So we "cheat" here to make empty = {0}
            ind[1] = {1}
            step = 2
            while (step < number_metaclasses):
                ind[step] = {step}
                step = step+1
                indatom = step
                for step2 in range(1,indatom - 1):
                    ind[step] = (ind[step2] | ind[indatom-1])
                    step = step+1
        out = np.zeros((number_metaclasses,number_metaclasses))

        for i in range(number_metaclasses):
            for j in range(number_metaclasses):
                out[i][j] = float(len(ind[i] & ind[j]))/float(len(ind[i] | ind[j]))

        # out[0,0] = 0
        return out

    
class Assignability:
    @staticmethod
    def from_matrix(similarity, fuzziness_coefficient=1):
        def assignability(mass):
            out = mass**fuzziness_coefficient
            out = np.dot(out, similarity)
            return out
        return assignability
    
    @staticmethod
    def cost_based(F, cautiousness_factor, fuzzifier_factor):
        def assignability(mass):
            out = mass**fuzzifier_factor
            out *= np.repeat(F.sum(axis=1).reshape(-1, 1), mass.shape[0], axis=1).T**cautiousness_factor
            out[:, 0] = mass[:, 0]**fuzzifier_factor 
            return out
        return assignability

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

        self.median_distance = np.median(self.distance_matrix)
        self.mean_distance = np.mean(self.distance_matrix)


    def get_silhouette(self, assignability_fn, outlier_coefficient=2):
        def sillhouette_fn(mass):
            # function to compute the average distance between points and metaclusters
            def a(assignabilit_matrix):
                out = np.dot(self.distance_matrix, assignabilit_matrix)
                out /= assignabilit_matrix.sum(axis=0)
                out[:, 0] = self.median_distance * outlier_coefficient / np.min(out[:, 1:], axis=1)
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
            sillhouette = (assignability_mat*s).sum(axis=1)/assignability_mat.sum(axis=1)

            return sillhouette.mean()

        return sillhouette_fn
        