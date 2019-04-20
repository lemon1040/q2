import numpy as np


class LabelShow():
    """
    根据用户索引，特征索引，聚类结果， 用户特征矩阵
    找出每组聚类的标签并显示
    """

    def __init__(self, index_category, cluster, array_user):
        self.index_category = index_category
        self.cluster = cluster
        self.array_user = array_user

    def __category_index(self):
        array_user, cluster = self.array_user, self.cluster
        hot_category_index = []
        for clu in cluster:
            clu_array = np.zeros(len(array_user[0, :]))
            for user_index in clu:
                clu_array += array_user[user_index, :]
            des = np.argsort(-clu_array)
            hot_category_index.append(list(des[0: min(int(0.01 * len(des)), 3)]))
        return hot_category_index

    def get_label(self):
        index_category = self.index_category
        hot_category_index = self.__category_index()
        hot_category = []
        for clu in hot_category_index:
            clu_array = []
            for category_index in clu:
                clu_array.append(list(index_category.keys())[category_index])
            hot_category.append(clu_array)
        return hot_category