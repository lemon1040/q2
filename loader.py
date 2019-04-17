import json


def load(path):
    with open(path, 'r') as f:
        dicts = json.load(f)
        return dicts


def save(path, dicts):
    with open(path, 'w') as f:
        json.dump(dicts, f)


if __name__ == '__main__':
    data = load('./data/user_business_223699.json')
    print(data)
