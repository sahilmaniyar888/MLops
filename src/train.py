from sklearn.model_selection import train_test_split
from joblib import dump

def train( X, y , model, config):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["train"]["test_size"], random_state=config["train"]["random_state"])


    model.fit(X_train, y_train)
    dump(model, config["output"]["model_path"])
    return model, X_test, y_test


