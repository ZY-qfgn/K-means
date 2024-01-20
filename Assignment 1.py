from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt

def initialization(data: np.ndarray, k: int) -> np.array:
    '''randomly initialize cluster centroids'''
    centroids = []
    index = []
    for i in range(1, k+1):
        ind = np.random.randint(1, len(data)+1)
        # initialization w/o repetition
        if ind not in index:
            index.append(ind)
            centroids.append(data[ind])
        else:
            continue

    return centroids


def get_distance(pt1: np.array, pt2: np.array) -> float:
  '''Calculate Euclidean distance between two points'''
  #or use pt1.ndim, is the same
  if pt1.shape != pt2.shape:
      print('pt1:', pt1, pt1.ndim, 'pt2:', pt2, pt2.ndim)
      raise ValueError("Two points have different dimensions.")

  return np.linalg.norm(pt1 - pt2)


def centroids_assignment(data: np.ndarray, centroids: np.array) -> dict:
    '''assign centroids for each data point, return a dictionary'''
    if len(centroids) == 0:
        raise ValueError("The centroids list is empty.")
    # initialize centroids_list
    centroids_list = {i : [] for i in range(len(centroids))}
    for point in data:
        min_ind = None
        min_dis = float('inf')
        # key of dic cannot be array type
        for ind, centroid in enumerate(centroids):
            dis = get_distance(point, centroid)
            # for min. dis
            if dis < min_dis:
                min_dis = dis
                min_ind = ind
        centroids_list[min_ind].append(point)

    return centroids_list


def move_centroids(centroids_list: dict) -> np.array:
    '''relocate the centroids by calculating the mean of points wrt that centroid'''
    new_centroids = []
    for ind, pts in centroids_list.items():
        new_centroids.append(np.mean(pts, axis=0))

    return new_centroids


def calculate_distortion(centroids: np.ndarray, centroids_list: dict) -> float:
    '''calculate the distortion cost of the k-means algorithm'''
    cost = 0
    num = 0
    for ind, sublist in centroids_list.items():
        centroid = centroids[ind]
        for point in sublist:
            cost += get_distance(point, centroid)**2
        num += len(sublist)

    return cost/num


def k_means(data: np.ndarray, k: int) -> [np.ndarray, dict]:
    centroids = np.array([])
    #new_centroids = np.array([])
    if len(data) == 0:
        raise ValueError("The dataset is empty.")

    new_centroids = initialization(data, k)
    centroids_list = {}
    while not np.array_equiv(centroids, new_centroids):
        # update centroids
        centroids = new_centroids
        centroids_list = centroids_assignment(data, centroids)
        new_centroids = move_centroids(centroids_list)

    return [new_centroids, centroids_list]

if __name__ == '__main__':
    # from sklearn import datasets
    row_data = load_breast_cancer()
    data, target = row_data.data, row_data.target
    #print(data.ndim)
    interval = range(2, 8)
    distortions = []
    for i in interval:
        centroids, corresponding_data = k_means(data, i)
        distortion = calculate_distortion(centroids, corresponding_data)
        distortions.append(distortion)
        #print('k:', i, '  distortion:', distortion(centroids, corresponding_data))
    for i in range(0,5):
        print('Slpoe change from k=', i+2, 'to k=', (i+3), ':', abs(distortions[i+1]-distortions[i])/distortions[i])

    plt.figure(figsize=(6,4))
    plt.plot(interval, distortions, 'o-')
    plt.xlabel('k value')
    plt.ylabel('Distortion')
    plt.title('Distortions achieved by κ-means with k range from 2 to 7')
    plt.scatter(interval[2], distortions[2], s=50, c='r', label='Elbow Point', zorder=5)
    plt.legend()
    plt.show()


