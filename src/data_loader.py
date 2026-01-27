from sklearn.datasets import load_iris


def load_data():
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return X, y
