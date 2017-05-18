import numpy as np


print "Starting K_means"
coordinates = np.loadtxt("data.txt")
centers = np.loadtxt("center.txt")
nloops = 10
eps = 1e-1
print "With coordinates {0} and centers {1}".format(coordinates, centers)

list_elem = [[]]*len(centers)

for i in xrange(nloops):
    distance_matrix = np.zeros((len(coordinates), len(centers)))
    for j in xrange(len(coordinates)):
        for k in xrange(len(centers)):
            distance_matrix[j][k] = (((coordinates[j][0]-centers[k][0])**-2) + (coordinates[j][1]-centers[k][1])**-2)
    distance_sum = np.sum(distance_matrix, axis=1)
    print "The distance matrix is {0}".format(distance_matrix)
    norm_matrix = distance_matrix/distance_sum[:, np.newaxis]
    print "The norm matrix is {0}".format(norm_matrix)
    for j in xrange(len(centers)):
        x , y = 0, 0
        norm_sum = 0
        for k in xrange(len(coordinates)):
            x, y = x + norm_matrix[k][j]*coordinates[k][0], y + norm_matrix[k][j]*coordinates[k][1]
            norm_sum += norm_matrix[k][j]
        centers[j] = (x/norm_sum, y/norm_sum)
    print "With new centers of {0}".format(centers)




