import math
import os
import logging
from sklearn import preprocessing
# from Distance.get_distance import GetDistance
import fastCluster.plot as plot
import numpy as np

path = os.getcwd() + r'\data' + '\\'
logger = logging.getLogger('data_pretreatment')


class FastSearchCluster(object):

    def __init__(self):
        self.distance = {}  # the dictionary that save all of the distance between two vectors
        self.max_dis = -1  # max distance
        self.min_dis = float('inf')  # min_distance
        self.data_size = 1  # the number of points
        self.result = []  # element: [rho, delta]
        self.rho_des_index = np.zeros(0)
        self.dc = 0
        self.master = np.zeros(0)  # delta point
        self.max_pos = -1  # the position where density is the max
        self.max_density = -1  # the density of max_pos
        self.gamma = []  # rho * delta
        self.gamma_des_index = []  # Index from big to small
        self.cluster_center = []  # cluster center

        self.cluster_temp = []  # mark the cluster_center for every point
        self.cluster = []  # the final cluster

    def __initCluster(self):
        """
        init cluster_temp for all the center point
        need:
            cluster_center
        """
        data_size, cluster_center = self.data_size, self.cluster_center
        self.cluster_temp = np.zeros(data_size, dtype=int)
        self.cluster_upper_bound = np.full(len(cluster_center), float('inf'), dtype=float)
        for center in cluster_center:
            self.cluster_temp[center] = center

    def load_dis_data(self, filename):
        """
        load data to memory
        init:
            distance
            data_size
            min_dis
            max_dis
            master
        """
        logger.info('load data')
        self.distance, self.data_size = {}, 1
        for line in open(path + filename, 'r'):
            x1, x2, d = line.strip().split(' ')
            x1, x2, d = int(x1), int(x2), float(d)
            self.data_size = max(x2 + 1, self.data_size)
            self.max_dis = max(self.max_dis, d)
            self.min_dis = min(self.min_dis, d)
            self.distance[(x1, x2)] = d
        self.master = np.zeros(self.data_size, dtype=int)
        logger.info('load accomplish')

    def get_dc(self, auto=False, percent=0.018):
        """
        select the distance ranked
        if not auto, we will choose the distance at 1.8% top position as dc
        :param
            auto
        :return
            dc
        need:
            data_size
            distance
            min_dis
            max_dis
        """
        data_size, distance = self.data_size, self.distance
        if not auto:
            position = int((data_size * (data_size + 1) / 2 - data_size) * percent)
            dc = sorted(distance.items(), key=lambda item: item[1])[position][1]
            logger.info("dc - " + str(dc))
            return dc
        else:
            min_range, max_range = self.min_dis, self.max_dis
            dc = (min_range + max_range) / 2
            while True:
                avg_rho_percent = sum([1 for d in distance.values() if d < dc]) / data_size ** 2 * 2
                if 0.01 <= avg_rho_percent <= 0.02:
                    break
                if avg_rho_percent < 0.01:
                    min_range = dc
                else:
                    max_range = dc
                dc = (min_range + max_range) / 2
                if max_range - min_range < 0.01:
                    break
            return dc

    def calculate_density(self, dc, cut_off=False):
        """
        calculate the density of each vector
        and get the max_pos
        :param
            dc
            cut_off
        need:
            data_size
            distance
        get:
            result[:, 0]
            max_pos
            max_density
            rho_des_index
        """
        data_size, distance = self.data_size, self.distance
        logger.info('calculate density begin')
        func = lambda dij, dc: math.exp(- (dij / dc) ** 2)
        if cut_off:
            func = lambda dij, dc: 1 if dij < dc else 0
        max_density = -1
        for index in range(data_size):
            density = 0
            for front in range(index):
                density += func(distance[(front, index)], dc)
            for later in range(index + 1, data_size):
                density += func(distance[(index, later)], dc)
            self.result.append([density, float("inf")])
            max_density = max(max_density, density)
            if max_density == density:
                self.max_pos = index
                self.max_density = max_density
        self.result = np.array(self.result)
        self.rho_des_index = np.argsort(-self.result[:, 0])
        logger.info('calculate density end')

    def calculate_delta(self):
        """
        calculate the delta of each vector
        save the delta point as master
        need:
            rho_des_index
            distance
        get:
            result[:, 1]
        """
        rho_des_index, distance, data_size = self.rho_des_index, self.distance, self.data_size
        self.result[rho_des_index[0]][1] = -1
        for i in range(1, data_size):
            for j in range(0, i):
                old_i, old_j = rho_des_index[i], rho_des_index[j]
                min_pos, max_pos = min(old_j, old_i), max(old_j, old_i)
                if distance[(min_pos, max_pos)] < self.result[old_i][1]:
                    self.result[old_i][1] = distance[(min_pos, max_pos)]
                    self.master[old_i] = old_j
        self.result[rho_des_index[0]][1] = max(self.result[:, 1])

    def calculate_gamma(self):
        """
        use the multiplication of normalized rho and delta as gamma to determine cluster center
        need:
            result[:, :]
        get:
            gamma
            gamma_des_index
        """
        result = self.result
        # scaler = preprocessing.StandardScaler()
        # train_minmax = scaler.fit_transform(result)
        # st_rho, st_delta = train_minmax[:, 0], train_minmax[:, 1]
        # self.gamma = (st_delta + st_rho) / 2
        self.gamma = result[:, 0] * result[:, 1]
        self.gamma_des_index = np.argsort(-self.gamma)

    def calculate_cluster_center(self, threshold):
        """
        Intercept a point with gamma greater than 0.2 as the cluster center
        need:
            gamma
        get:
            cluster_center
        """
        gamma = self.gamma
        self.cluster_center = np.where(gamma >= threshold)[0]

    def pre_data(self, filename, autoDc=False):
        self.load_dis_data(filename)
        self.dc = self.get_dc(auto=autoDc)
        self.calculate_density(self.dc)
        self.calculate_delta()
        self.calculate_gamma()

    def get_cluster(self):
        self.__initCluster()
        rho_des_index, cluster_center = self.rho_des_index, self.cluster_center
        for index in rho_des_index:
            if index not in cluster_center:
                self.cluster_temp[index] = self.cluster_temp[self.master[index]]
        for index in range(len(cluster_center)):
            self.cluster.append(np.where(self.cluster_temp == cluster_center[index])[0])
        for index, cluster in enumerate(self.cluster):
            for pos in cluster:
                self.cluster_temp[pos] = index
        # self.calculate_upper_bound(self.dc)
        # self.get_hole()


def plot_rho_delta(result):
    # plot rho and delta
    plot.plot_diagram(result[:, 0], result[:, 1], 'rho', 'delta', 'Decision Graph')


def plot_gamma(gamma, data_size):
    # plot gamma diagram to get cluster center
    des_gamma = sorted(gamma, reverse=True)
    plot.plot_diagram(np.arange(int(data_size)), des_gamma[:int(data_size)], 'x', 'gamma', 'gamma diagram')


def save_result(filename, vectors, cluster_temp):
    out = open(path + filename, 'w')
    for index, vector in enumerate(vectors):
        out.write(str(vector[0]) + ' ' + str(vector[1]) + ' ' + str(cluster_temp[index]) + '\n')
    out.close()


if __name__ == '__main__':

    pre = FastSearchCluster()
    pre.pre_data('dis_cos_directUser.txt')

    plot_gamma(pre.gamma, pre.data_size / 7)
    # pre = FastSearchCluster()
    # pre.pre_data('output.txt')
    # pre.calculate_cluster_center(2.9)
    # pre.get_cluster()
    #
    # plot_rho_delta(pre.result)
    # plot_gamma(pre.gamma, pre.data_size)

    # plot the result
    # builder = GetDistance()
    # builder.load('Aggregation.txt')
    # plot.plot_cluster('x', 'y', 'cluster', pre.cluster, builder.vectors)
    # save_result('Task1.csv', builder.vectors, pre.cluster_temp)

