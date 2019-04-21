####### 装载数据 并生成标签索引#########################
#%% 
##设置禁用标签
banned_category = []

## 装载数据
from loader import index_builder
builder = index_builder(banned_category)
array_user, index_category, index_user = builder.build_index()
array_user_origin = array_user.copy()
del builder
#%%
import numpy as np
category = np.zeros(len(array_user_origin[0]))
for tmp in array_user_origin:
    category += np.array(tmp, dtype=int)
need_index = np.where(category>2000)[0]

## 手动降维， 将出现次数低于2000次的标签删除
for index, row in enumerate(array_user):
    array_user[index] = [x for index, x in enumerate(row) if index in need_index]

## PCA降维
# from sklearn.decomposition import PCA
# pac = PCA(n_components=0.95)
# array_user = pac.fit_transform(array_user)
## 数据标准化
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
array_user = scaler.fit_transform(array_user)
array_user_origin = scaler.fit_transform(array_user_origin)
######################################################################

#%%
## 余弦距离计算函数
def get_cos_dis(array):
    from sklearn.metrics.pairwise import pairwise_distances
    dis = pairwise_distances(array,metric="cosine")
    return dis

#%%
## 获得余弦距离
dis = get_cos_dis(array_user)

#%%
## 保存余弦距离
import loader
loader.save_dis('./data/dis_cos_directUser.txt', dis)

#%%
del dis



######## Kmeans 聚类 ###############################
#%%
from kmeansCluster import *

## Kmeans 聚类 类个数 + 特征矩阵
k = 4
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

## 显示聚类结果
from labelShow import LabelShow
show = LabelShow(index_category, cluster, array_user_origin)
hot_category = show.get_label()
print(hot_category)

#%%
for index, user in enumerate(array_user_origin):
    array_user_origin[index] = np.array(user)
array_user_origin = np.array(array_user_origin)
#####################################################



######## 2014 sci聚类 ###############################
#%%
from fastCluster.fastSearchCluster import FastSearchCluster
pre = FastSearchCluster()
pre.pre_data('dis_cos_directUser.txt')

#%%
## 打印出gamma图
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
des_gamma = sorted(pre.gamma, reverse=True)
for index, i in enumerate(des_gamma[:1000]):
    plt.scatter(index, i)
plt.title("gamma diagram")
plt.show()
plt.savefig('./data/img/gamma_diagram.png')

#%%
pre.calculate_cluster_center(500)
pre.get_cluster()
#%%
## 显示聚类结果
from labelShow import LabelShow
show = LabelShow(index_category, pre.cluster, array_user_origin)
hot_category = show.get_label()
for clu_labels in hot_category:
    print(clu_labels)

#%%
## 评估聚类好坏
from sklearn import metrics
silhouette_score = metrics.silhouette_score(array_user, pre.cluster_temp, metric='cosine')
#####################################################



######## Hierarchical 聚类###########################

#%%
##执行层次聚类
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='average')
labels = model.fit(array_user)

#%%
## 得到聚类分组
import numpy as np
cluster = []
for i in range(labels.n_clusters):
    tmp = []
    for ele in np.argwhere(labels.labels_ == i)[:, 0]:
        tmp.append(ele)
    cluster.append(tmp)

## 显示聚类结果
from labelShow import LabelShow
show = LabelShow(index_category, cluster, array_user_origin)
hot_category = show.get_label()
for clu_labels in hot_category:
    print(clu_labels)

#%%
## 选取K值
distortions = []
range_list = [2,3,4,5,6,7,8]
for i in range_list:
    model = AgglomerativeClustering(n_clusters=i, affinity='cosine', linkage='average')
    model.fit(array_user)
    silhouette_score = metrics.silhouette_score(array_user, model.labels_, metric='cosine')
    distortions.append(silhouette_score)
measure_k(range_list, distortions)

#####################################################



######## Spectral 聚类################################
#%%
## 执行谱聚类
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',n_neighbors=11)
labels = model.fit(array_user)

#%%
## 评估执行结果
from sklearn import metrics
silhouette_score = metrics.silhouette_score(array_user, model.labels_, metric='euclidean')
print(silhouette_score)

#%%
## 得到聚类分组
import numpy as np
cluster = []
for i in range(labels.n_clusters):
    tmp = []
    for ele in np.argwhere(labels.labels_ == i)[:, 0]:
        tmp.append(ele)
    cluster.append(tmp)

## 显示聚类结果
from labelShow import LabelShow
show = LabelShow(index_category, cluster, array_user_origin)
hot_category = show.get_label()
for clu_labels in hot_category:
    print(clu_labels)


#%%
## affinity='rbf' 选取gamma值
distortions = []
range_list = [0.01, 0.05, 0.1, 0.2, 0.3]
for i in range_list:
    model = SpectralClustering(n_clusters=4, affinity='rbf', gamma=i)
    model.fit(array_user)
    silhouette_score = metrics.silhouette_score(array_user, model.labels_, metric='euclidean')
    distortions.append(silhouette_score)

#%%
from kmeansCluster import measure_k
measure_k(range_list, distortions)

#%%
# affinity='nearest_neighbors' 选取n_neighbors值
distortions = []
range_list = [ 10, 20, 30, 40, 50]
for i in range_list:
    model = SpectralClustering(n_clusters=4, affinity='nearest_neighbors',n_neighbors=i)
    model.fit(array_user)
    silhouette_score = metrics.silhouette_score(array_user, model.labels_, metric='euclidean')
    distortions.append(silhouette_score)
######################################################



######## DBSCAN 聚类################################
#%%
## 执行DBSCAN
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.4, min_samples=13, metric='cosine').fit(array_user)
n_clusters_ = np.max(labels.labels_) + 1
print(n_clusters_)
#%%
## 评估执行结果
from sklearn import metrics
silhouette_score = metrics.silhouette_score(array_user, labels.labels_, metric='cosine')
print(silhouette_score)

#%%
## 得到聚类分组
import numpy as np
cluster = []
for i in range(n_clusters_):
    tmp = []
    for ele in np.argwhere(labels.labels_ == i)[:, 0]:
        tmp.append(ele)
    cluster.append(tmp)
#%%
## 显示聚类结果
from labelShow import LabelShow
show = LabelShow(index_category, cluster, array_user_origin)
hot_category = show.get_label()
for clu_labels in hot_category:
    print(clu_labels)


#%%
## 选取eps 和 min_samples
distortions = []
range_list = []
for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for min_samples in range(5, 30):
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(array_user)
        n_clusters_ = np.max(db.labels_) + 1
        if n_clusters_ in [3, 4, 5]:
            range_list.append([eps, min_samples])
            distortions.append(metrics.silhouette_score(array_user, db.labels_, metric='cosine'))


#%%
from kmeansCluster import measure_k
measure_k(range_list, distortions)

######################################################



########## EM-GMM ######################################
#%%
from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=4)
labels = gm.fit_predict(array_user)
n_clusters_ = np.max(labels) + 1
print(n_clusters_)

#%%
## 评估执行结果
from sklearn import metrics
silhouette_score = metrics.silhouette_score(array_user, labels)
print(silhouette_score)

#%%
## 得到聚类分组
import numpy as np
cluster = []
for i in range(n_clusters_):
    tmp = []
    for ele in np.argwhere(labels == i)[:, 0]:
        tmp.append(ele)
    cluster.append(tmp)
#%%
## 显示聚类结果
from labelShow import LabelShow
show = LabelShow(index_category, cluster, array_user_origin)
hot_category = show.get_label()
for clu_labels in hot_category:
    print(clu_labels)

#%%
## 选取K值
distortions = []
range_list = [2,3,4,5,6]
for i in range_list:
    model = gm = GaussianMixture(n_components=i)
    labels = gm.fit_predict(array_user)
    silhouette_score = metrics.silhouette_score(array_user, labels)
    distortions.append(silhouette_score)
measure_k(range_list, distortions)
##################################################################