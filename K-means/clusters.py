import numpy as np


def generate_data(num_point, cluster_center):
    dat_x = np.random.randn(num_point, 2)
    cluster = np.array(cluster_center)
    return (dat_x+cluster_center)

def construct_cluster(num_points, clusters):
    """Constructs num_points number of points at the specified cluster locations. Clusters
    should be a list of tuples of x and y coordinates."""
    data = generate_data(num_points, clusters[0])
    for i in xrange(1, len(clusters)):
        new_dat = generate_data(num_points, clusters[i])
        data = np.vstack((data, new_dat))
    return data


if __name__ == "__main__":
    data = construct_cluster(10, [(2,1), (5,7), (-4, -4)])
    np.savetxt("data.txt", data)
    np.savetxt("center.txt", np.random.randn(3, 2)*4)


