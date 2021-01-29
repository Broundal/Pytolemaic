import numpy
from sklearn.neighbors import NearestNeighbors


class Clustering():

    def __init__(self, n_neighbors=50, radii_inc_percentile=3, looking_from_outside=True):
        self.centers = []
        self.centers_inds = []
        self.clusters = {}
        self.clusters_inds = {}
        self.indices_to_cluster = None
        self.n_neighbors = n_neighbors
        self.radii_inc_percentile = radii_inc_percentile
        self.looking_from_outside = looking_from_outside

    def _get_knn(self, x, prev_nn, radius, cached_n_neighbors=None):
        n_neighbors = cached_n_neighbors or min(self.n_neighbors, len(x))
        knn_distances, knn_indices = prev_nn.kneighbors(x, n_neighbors=min(self.n_neighbors, len(x)))
        while numpy.min(knn_distances[:, -1]) < radius and n_neighbors < len(x):
            knn_distances, knn_indices = prev_nn.kneighbors(x, n_neighbors=min(self.n_neighbors, len(x)))
            n_neighbors = min(2 * n_neighbors, len(x))
            print(n_neighbors)

        return knn_distances, knn_indices, n_neighbors

    def fit(self, X):
        nn = NearestNeighbors()
        x = X

        orig_indices = numpy.arange(len(X))
        self.indices_to_cluster = -1 * numpy.ones(len(X)).astype(int)
        while len(x) > 0:
            nn.fit(x)
            cached_n_neighbors = None

            # detect new center, and add it.
            knn_distances, knn_indices = nn.kneighbors(x, n_neighbors=min(self.n_neighbors, len(x)))

            ind = numpy.argmin(knn_distances[:, -1])
            radius = knn_distances[ind, -1]

            orig_ind = orig_indices[ind]
            self.centers_inds.append(orig_ind)
            self.centers.append(X[orig_ind, :])

            cluster_id = len(self.centers) - 1

            # add points to cluster
            self.clusters[cluster_id] = []
            self.clusters_inds[cluster_id] = []

            candidate_points = numpy.ones(len(x)).astype(bool)
            inds_found = [ind]

            while len(inds_found) > 0:
                # extend list of points in cluster
                self.clusters_inds[cluster_id].extend(orig_indices[inds_found])

                # mark found points
                candidate_points[inds_found] = False

                # update radius
                if len(self.clusters_inds[cluster_id]) > self.n_neighbors:
                    dists, inds_ = nn.kneighbors(X[self.clusters_inds[cluster_id], :], n_neighbors=self.n_neighbors)
                    in_cluster = [ind in self.clusters_inds[cluster_id] for ind in inds_[:, -1]]  # inefficient
                    in_cluster = numpy.array(in_cluster)
                    dists = dists[in_cluster, -1]
                    if len(dists) > 0:
                        radius_ = numpy.percentile(dists, self.radii_inc_percentile)
                        # print(cluster_id, radius_, len(self.clusters_inds[cluster_id]))
                        radius = radius_

                # locate additional points
                new_inds = knn_indices[inds_found, :][(knn_distances[inds_found, :] <= radius)]  # !!!!!!!!!!!
                new_inds = new_inds[candidate_points[new_inds]]
                inds_found = list(set(new_inds.tolist()))

                if len(inds_found) == 0 and self.looking_from_outside:
                    knd = knn_distances[candidate_points, :]
                    kni = knn_indices[candidate_points, :]
                    not_in_cluster = numpy.arange(len(candidate_points))[candidate_points]
                    for ind in not_in_cluster:
                        kni[kni == ind] = -1
                    kni[kni >= 0] = 1
                    kni[kni < 0] = 0
                    knd = knd <= radius
                    kn = kni * knd.astype(int)
                    is_close = numpy.sum(kn, axis=1) > 2
                    inds_found = not_in_cluster[is_close]
                    if len(inds_found) > 0:
                        print('added looking_from_outside:', len(inds_found))

            self.clusters_inds[cluster_id] = list(set(self.clusters_inds[cluster_id]))
            self.clusters[cluster_id] = X[self.clusters_inds[cluster_id], :]
            self.indices_to_cluster[self.clusters_inds[cluster_id]] = cluster_id
            x = x[candidate_points, :]
            orig_indices = orig_indices[candidate_points]

        self.nn = NearestNeighbors()
        self.nn.fit(X)
        print(self.centers)

    def predict(self, X):

        inds = self.nn.kneighbors(X, 1, return_distance=False)
        return self.indices_to_cluster[inds]
