from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

def get_Kmeanscluster(k, array_user):
    method = KMeans(n_clusters=k, random_state=9, algorithm='full')
    kmeans_cluster = method.fit(array_user)
    return kmeans_cluster

def SSEmeasure(array_user, range_list):
    """
    手肘法评测选取一个合适的k值
    """
    distortions = []
    for i in range_list:
        km = KMeans(n_clusters=i, random_state=9, algorithm='full')
        km.fit(array_user)
        #获取K-means算法的SSE
        distortions.append(km.inertia_)
    return distortions

def SCmeasure(array_user, range_list):
    """
    轮廓系数法选取合适的K值
    """
    distortions = []
    for i in range_list:
        km = KMeans(n_clusters=i, random_state=9, algorithm='full')
        km.fit(array_user)
        silhouette_score = metrics.silhouette_score(array_user, km.labels_, metric='euclidean')
        distortions.append(silhouette_score)
    return distortions

## 绘制KMeans K的聚类效果变化图
def measure_k(ranklist, distortions):
    plt.plot(ranklist,distortions,marker="o")
    plt.xlabel("簇数量")
    plt.ylabel("簇内误方差(SSE)")
    plt.show()