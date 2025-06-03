import mlflow
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import dagshub
dagshub.init(repo_owner='vashisthrahul13', repo_name='mlflow-dagshub-learn', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/vashisthrahul13/mlflow-dagshub-learn.mlflow")

print(mlflow.get_tracking_uri) #Defines where MLflow logs and retrieves runs


iris = load_iris()
X= iris.data
Y = iris.target

#train test split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#define parameter
max_depth = 10

#autlogging using mlflow
mlflow.autolog()

mlflow.set_experiment('iris-dt') # will use iris-dt if already create or will create new experiment and then use

#start mlflow run
    #with mlflow.start_run(run_name = 'exp0_run2') --> method to define run name
    #with mlflow.start_run(experiment_id=): -> if already exp created 
with mlflow.start_run():

    #train random forest
    dt  = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(x_train,y_train)

    #make predictions
    y_pred = dt.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt. figure(figsize=(6,6))
    sns. heatmap (cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel( 'Actual')
    plt. xlabel ( 'Predicted')
    plt. title( 'Confusion Matrix')
    # Save the plot as an artifact
    plt. savefig ("confusion_matrix.png")

    #log code
    mlflow.log_artifact(__file__)


    #add tags
    mlflow.set_tag('author','rahul')
    mlflow.set_tag('model','decision tree')

    