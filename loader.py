import json


def load(path):
    with open(path, 'r') as f:
        dicts = json.load(f)
        return dicts

def save(path, dicts):
    with open(path, 'w') as f:
        json.dump(dicts, f)


def save_dis(path, array):
    out = open(path, 'w')
    for i in range(len(array[:, 0])):
        for j in range(i + 1, len(array[0, :])):
            out.write(str(i) + ' ' + str(j) + ' ' + str(array[i, j]) + '\n')
    out.close()


class index_builder():
	"""
	构建用户和用户特征列表的对应 index_user
	构建标签和用户特征向量维度的对应 index_category
	"""

	def __init__(self, banned_category):
		self.banned_category = banned_category  ## 被禁止的标签
		self.data_user = {}
		self.data_business = {}

	def __load_data__(self):
		self.data_user = load('./data/userMini.json')
		self.data_business = load('./data/businessMini.json')
	
	def build_index(self):
		self.__load_data__()
		index_category = self.build_category_index()
		index_user, array_user = self.build_user_index(index_category)
		return array_user, index_category, index_user

	def build_category_index(self):
		business, banned_category = self.data_business, self.banned_category
		index_category = {}
		for v in business.values():
			for category in v['categories']:
				if category not in index_category.keys() and category not in banned_category:
					index_category[category] = 0
		for index, k in enumerate(index_category.keys()):
			index_category[k] = index
		return index_category

	def build_user_index(self, index_category):
		user, business, banned_category = self.data_user, self.data_business, self.banned_category
		index_user = {}
		array_user = []
		for k, v in user.items():
			index_user[k] = 0
			tmp = [0 for i in range(len(index_category))]
			for businessId in v:
				for category in business[businessId]['categories']:
					if category not in banned_category:
						tmp[index_category[category]] += 1
			array_user.append(tmp)
		for index, k in enumerate(index_user.keys()):
			index_user[k] = index
		return index_user, array_user


if __name__ == '__main__':
    data = load('./data/user_business_223699.json')
    print(data)
