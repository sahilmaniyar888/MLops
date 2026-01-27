from sklearn.ensemble import RandomForestClassifier

def build_model(config):
    params = config["model"]["params"]
    model = RandomForestClassifier(**params)
    return model