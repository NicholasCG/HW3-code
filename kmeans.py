import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import os
import glob
from time import time

# euclidean distance
@jit(nopython=True)
def euclidean(x, y):
    return np.linalg.norm(x - y)

# cosine distance
@jit(nopython=True)
def cosine(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# jaccard distance
@jit(nopython=True)
def jaccard(x, y):
    return 1 - np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))

class KMeans:
    def __init__(self, x, y, k=10, max_iter=200):
        self.x = x
        self.y = y

        self.k = len(np.unique(y))
        self.max_iter = max_iter

        self.init_centroids = x[np.random.choice(len(x), self.k, replace=False)]

    def fit(self):
        print("\nTermination condition: centroids don't move")
        for loss in [euclidean, cosine, jaccard]:
            self.loss = loss
            self._fit('centroids')

        print("\nTermination condition: SSE increases")
        for loss in [euclidean, cosine, jaccard]:
            self.loss = loss
            self._fit('SSE')

        print("\nTermination condition: max iteration")
        for loss in [euclidean, cosine, jaccard]:
            self.loss = loss
            self._fit('max_iter')

        print("\nTermination condition: All three")
        for loss in [euclidean, cosine, jaccard]:
            self.loss = loss
            self._fit('all')

    def _fit(self, condition):
        print("\nKMeans with {} loss function".format(self.loss.__name__))

        centroids = self.init_centroids
        old_centroids = None

        old_sse = float('inf')
        sse = float('inf')
        
        sses = []
        i = 0

        if condition == 'centroids':
            start_time = time()
            while (old_centroids is None or not np.array_equal(old_centroids, centroids)):

                old_centroids = centroids

                assigned_centroids = self._assign_centroid(self.x, centroids)

                centroids = self._update_centroid(self.x, assigned_centroids)

                sse = self._SSE(self.x, centroids, assigned_centroids)

                sses.append(sse)
                i += 1
            end_time = time()

        elif condition == 'SSE':
            start_time = time()
            while old_sse == float('inf') or sse < old_sse:
                old_sse = sse

                assigned_centroids = self._assign_centroid(self.x, centroids)

                centroids = self._update_centroid(self.x, assigned_centroids)

                sse = self._SSE(self.x, centroids, assigned_centroids)

                sses.append(sse)
                i += 1
            end_time = time()

        elif condition == 'max_iter':
            start_time = time()
            for i in range(self.max_iter):

                assigned_centroids = self._assign_centroid(self.x, centroids)

                centroids = self._update_centroid(self.x, assigned_centroids)

                sse = self._SSE(self.x, centroids, assigned_centroids)
                sses.append(sse)

            end_time = time()

        elif condition == 'all':
            start_time = time()
            while (old_centroids is None or not np.array_equal(old_centroids, centroids)) and (old_sse == float('inf') or sse < old_sse) and i < self.max_iter:
                old_centroids = centroids
                old_sse = sse

                assigned_centroids = self._assign_centroid(self.x, centroids)

                centroids = self._update_centroid(self.x, assigned_centroids)

                sse = self._SSE(self.x, centroids, assigned_centroids)
                sses.append(sse)

                i += 1
            end_time = time()

        print("Final SSE: {}".format(sse))
        # clear plot
        # plt.clf()
        plt.figure(figsize=(8, 5))
        # plot the sum of squared error
        plt.plot(sses)
        plt.xlabel('Iteration')
        plt.ylabel('SSE')
        plt.title("KMeans, {} loss, {} stop".format(self.loss.__name__, condition))

        assigned_centroids = self._assign_centroid(self.x, centroids)

        accuracy = self.accuracy(y, assigned_centroids)

        # display the final SSE, accuracy, time, and # of iterations on the plot to the right
        plt.figtext(0.6, 0.7, f'Final SSE: {sse:0.2f} \nAccuracy: {(accuracy * 100):0.2f}% \nTime: {(end_time - start_time):0.2f} sec \nIterations: {i}', wrap=True, horizontalalignment='left', fontsize=12)
        
        # save the plot
        # plt.show()
        plt.savefig(f'kmeans_{self.loss.__name__}_{condition}.png')


    def _SSE(self, x, centroids, assigned_centroids):
        sse = 0
        for i in range(self.k):
            if len(assigned_centroids[i]) > 0:
                cluster_data = x[assigned_centroids[i]]
                centroid = centroids[i]
                sse += np.sum(np.square(self.loss(cluster_data, centroid)))
        
        return sse

    def _assign_centroid(self, x, centroids):
        # assign each data point to the nearest centroid
        # return a list of assigned centroids for each data point
        assigned_centroids = [[] for _ in range(self.k)]

        distances = np.zeros((len(x), self.k))
        for i in range(self.k):
            distances[:, i] = np.apply_along_axis(self.loss, 1, x, centroids[i])

        clusters = np.argmin(distances, axis=1)

        for i, cluster in enumerate(clusters):
            assigned_centroids[cluster].append(i)

        # print('Assigned centroids: {}'.format(assigned_centroids))
        return assigned_centroids
    
    def _update_centroid(self, x, assigned_centroids):
        # update the centroids by taking the mean of all data points assigned to it
        # return a list of new centroids
        new_centroids = []
        for i in range(self.k):
            new_centroids.append(np.mean(x[assigned_centroids[i]], axis=0))

        return new_centroids
    
    def accuracy(self, y, assigned_centroids):
        # return the accuracy of the clustering
        
        num_correct = 0

        # for each cluster, find the most common label
        for i in range(self.k):
            bins = np.bincount(y[assigned_centroids[i]])
            label_i = np.argmax(bins)
            num_correct += bins[label_i]

        return num_correct / len(y)
if __name__ == '__main__':
    # delete all old images
    for file in glob.glob('*.png'):
        os.remove(file)

    # test each three loss functions using data.csv and label.csv
    x = pd.read_csv('data.csv', header=None).to_numpy(dtype=np.float64)

    # enfroce that x is contiguous in memory
    x = np.ascontiguousarray(x)

    y = pd.read_csv('label.csv', header=None).to_numpy().reshape(-1,)

    k = len(np.unique(y))
    print(f'Number of clusters in ground truth: {k}')

    kmeans = KMeans(x, y, k = k)
    kmeans.fit()
