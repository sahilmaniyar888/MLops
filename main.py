import yaml
from src import data_loader , model , train , evaluate

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))

    x,y = data_loader.load_data()
    clf = model.build_model()
    clf , X_test , y_test = train.train(x,y,clf,config)
    evaluate.evaluate(clf,X_test,y_test)
    
    print("Training and evaluation completed successfully!")