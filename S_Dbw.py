import numpy as np

class S_Dbw():
    def __init__(self,data,data_cluster,cluster_centroids_):
        """
        data --> raw data
        data_cluster --> The category that represents each piece of data(the number of category should begin 0)
        cluster_centroids_ --> the center_id of each cluster's center
        """
        self.data = data
        self.data_cluster = data_cluster
        self.cluster_centroids_ = cluster_centroids_

        # cluster_centroids_ 是一个 array, 给出类数k
        self.k = cluster_centroids_.shape[0]
        self.stdev = 0                # stdev 的初始化
        # 对每个类别标记进行循环，如：0，1，2，...
        # 该循环计算的是下面公式里根号里的内容：
        for i in range(self.k):
            # 计算某类别下所有样本各自的全部特征值的方差：
            #（vector，shape为样本的个数，相当于下面公式里的 signma）
            std_matrix_i = np.std(data[self.data_cluster == i],axis=0)
            # 求和
            self.stdev += np.sqrt(np.dot(std_matrix_i.T,std_matrix_i))
        self.stdev = np.sqrt(self.stdev)/self.k # 取平均


    def density(self,density_list=[]):
        """
        compute the density of one or two cluster(depend on density_list)
        变量 density_list 将作为此函数的内部列表，其取值范围是0,1,2,... ，元素个数是聚类类别数目
        """
        density = 0
        if len(density_list) == 2:    # 当考虑两个聚类类别时候，给出中心点位置
            center_v = (self.cluster_centroids_[density_list[0]] +self.cluster_centroids_[density_list[1]])/2
        else:                         # 当只考虑某一个聚类类别的时候，给出中心点位置
            center_v = self.cluster_centroids_[density_list[0]]
        for i in density_list:
            temp = self.data[self.data_cluster == i]
            for j in temp:    # np.linalg.norm 是求范数(order=2)
                if np.linalg.norm(j - center_v) <= self.stdev:
                    density += 1
        return density


    def Dens_bw(self):
        density_list = []
        result = 0
        # 下面的变量 density_list 列表将会算出每个对应单类的密度值。
        for i in range(self.k):
            density_list.append(self.density(density_list=[i])) # i 是循环类别标签
        # 开始循环排列
        for i in range(self.k):
            for j in range(self.k):
                if i==j:
                    continue
                result += self.density([i,j])/max(density_list[i],density_list[j])
        return result/(self.k*(self.k-1))

    def Scat(self):
        # 分母部分：
        sigma_s = np.std(self.data,axis=0)
        sigma_s_2norm = np.sqrt(np.dot(sigma_s.T,sigma_s))

        # 分子部分：
        sum_sigma_2norm = 0
        for i in range(self.k):
            matrix_data_i = self.data[self.data_cluster == i]
            sigma_i = np.std(matrix_data_i,axis=0)
            sum_sigma_2norm += np.sqrt(np.dot(sigma_i.T,sigma_i))
        return sum_sigma_2norm/(sigma_s_2norm*self.k)


    def S_Dbw_result(self):
        """
        compute the final result
        """
        return self.Dens_bw()+self.Scat()

#just for tests
#data = np.array([[1,2,1],[0,1,4],[3,3,3],[2,2,2]])
#data_cluster = np.array([1,0,1,2]) # The category represents each piece of data belongs
#centers_id = np.array([1,0,3]) # the cluster's num is 3

#a = S_Dbw(data,data_cluster,centers_id)
#print(a.S_Dbw_result())


# 例子
import S_Dbw as sdbw
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin

#import matplotlib.pyplot as plt

np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(X)

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

KS = sdbw.S_Dbw(X, k_means_labels, k_means_cluster_centers)
print(KS.S_Dbw_result())