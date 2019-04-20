
#%%
banned_category = []

#%%
from loader import index_builder
builder = index_builder(banned_category)
array_user, index_category, index_user = builder.build_index()

#%%
del builder


#%%
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
array_user = scaler.fit_transform(array_user)


#%%
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
array_user = scaler.fit_transform(array_user)


#%%
## 余弦相似度的距离
def get_cos_dis(array):
    from sklearn.metrics.pairwise import pairwise_distances
    dis = pairwise_distances(array,metric="cosine")
    return dis

#%%
dis = get_cos_dis(array_user)

#%%
import loader
loader.save_dis('./data/dis_cos_directUser.txt', dis)

#%%
del dis



######## Kmeans 聚类 ###############################
#%%
from kmeansCluster import *
## Kmeans 聚类 类个数 + 特征矩阵
k = 5
kmeans_cluster = get_Kmeanscluster(k, array_user)

#%%
## 手肘法评测聚类效果
clusters = [i for i in range(1, 11)]
distortions = SSEmeasure(array_user, clusters)

#%%
## 轮廓系数评测聚类效果
clusters = [2,3,4,5,8, 10]
distortions = SCmeasure(array_user, clusters)

#%%
## 显示评价结果
get_ipython().run_line_magic('matplotlib', 'inline')
measure_k(clusters, distortions)

#%%
## 得到聚类分组
import numpy as np
cluster = []
for i in range(kmeans_cluster.n_clusters):
    tmp = []
    for ele in np.argwhere(kmeans_cluster.labels_ == i)[:, 0]:
        tmp.append(ele)
    cluster.append(tmp)

#%%
## 显示聚类结果
from labelShow import LabelShow
show = LabelShow(index_category, cluster, array_user)
hot_category = show.get_label()
print(hot_category)
#####################################################



######## 2014 sci聚类 ###############################
#%%
from fastCluster.fastSearchCluster import FastSearchCluster
pre = FastSearchCluster()
pre.pre_data('dis_cos_directUser.txt')

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
from fastCluster.fastSearchCluster import plot_gamma
plot_gamma(pre.gamma, pre.data_size / 7)

#%%
import matplotlib.pyplot as plt
des_gamma = sorted(pre.gamma, reverse=True)
for index, i in enumerate(des_gamma[:1000]):
    plt.scatter(index, i)
plt.title("gamma diagram")
plt.show()
plt.savefig('./data/img/gamma_diagram.png')

#%%
pre.calculate_cluster_center(1460)

#%%
pre.get_cluster()
#####################################################





