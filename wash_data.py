import loader

class WashData:
    """
    选取满足条件的用户进行聚类， 并相应地获得用户对应的商家数据
    """
    def __init__(self):
        self.data_user = {}
        self.data_business = {}

    def load(self, u_path, b_path):
        self.data_user = loader.load(u_path)
        self.data_business = loader.load(b_path)

    def save(self, u_path, b_path):
        loader.save(u_path, self.data_user)
        loader.save(b_path, self.data_business)

    def wash(self, need_arr, lessUserFunc, *args):
        self.data_user = lessUserFunc(self.data_user, args)
        self.data_business = self.__used_business(need_arr)

    def __used_business(self, need_arr):
        data_user, data_business = self.data_user, self.data_business
        data_miniBusiness = {}
        for v in data_user.values():
            for businessId in v:
                if businessId not in data_miniBusiness.keys():
                    data_miniBusiness[businessId] = data_business[businessId]
        for businessId, v in data_miniBusiness.items():
            dicts = {}
            for k in v.keys():
                if k in need_arr:
                    dicts[k] = v[k]
            data_miniBusiness[businessId] = dicts
        return data_miniBusiness


def FrequentUser(user_dict, num):
    miniUser = {}
    for k, v in user_dict.items():
        if len(v) >= num:
            miniUser[k] = v
    return miniUser


if __name__ == "__main__":
    need_arr = ['categories', 'city']
    washer = WashData()
    washer.load('./data/user_business_223699.json', './data/business_163665.json')
    washer.wash(need_arr, FrequentUser, 50)
    washer.save('./data/userMini.json', './data/businessMini.json')
