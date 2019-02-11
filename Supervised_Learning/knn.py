import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import time


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import model_selection
from sklearn.grid_search import GridSearchCV

def getData():
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')
    x, y = df.values[:, 0:len(df.columns)-1] , df.values[:, len(df.columns) - 1]
    return x, y 
   
def getPokerData():
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data')
    x, y = df.values[:, 0:len(df.columns)-1] , df.values[:, len(df.columns) - 1]
    return x, y 

def runKNN(x, y):
    knn = KNeighborsClassifier()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.3, random_state = 100)
    k_range = [i for i in range(1, 15)]
    param_grid = dict(n_neighbors=k_range)    
    grid = GridSearchCV(knn, param_grid, cv = 2, scoring = 'accuracy')
    start = time.time()
    grid.fit(x_train, y_train)
    end = time.time() - start
    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
    graph_data = [k_range, grid_mean_scores]
    return grid.best_estimator_, grid.best_score_, [x_test, y_test], graph_data, end

def getMetrics(dataset):
    if dataset == "Poker Hand":
        x, y = getPokerData()
    else:
        x, y = getData() 
    print("k-Nearest Neighbors Model Metrics for " + dataset + ":")
    clf, training_score, testing_data, graph_data, elapsed_time = runKNN(x, y)
    print("Training time: {0:.2f}s".format(elapsed_time))
    start_time = time.time()
    resultData = clf.predict(testing_data[0])
    elapsed_time = time.time() - start_time
    print("Testing time: {0:.2f}s".format(elapsed_time))
    print("Training score: {0:.2f}%".format(training_score * 100))
    print("Confusion Matrix: \n",
      confusion_matrix(testing_data[1], resultData))
    a_confusion_matrix = confusion_matrix(testing_data[1], resultData)
    df_cm = pd.DataFrame(a_confusion_matrix, index =[i for i in range(0, len(a_confusion_matrix[0]))], columns = [i for i in range(0, len(a_confusion_matrix))])
    plt.figure(figsize = (6, 4))
    sn.heatmap(df_cm, annot=True)
    print("Accuracy Score: {0:.2f}%\n\n".format(accuracy_score(testing_data[1], resultData) * 100))
    
    fig = plt.figure(200)
    ax1 = plt.subplot(211)
    ax1.plot(graph_data[0], graph_data[1])
    ax1.set_xlabel("k-Nearest Neighbor")
    ax1.set_ylabel("Cross Validation Score")
    ax1.set_title("CV Scores in correlation with KNN")
    fig.tight_layout()
    plt.show()
    
    print("\n\n")
    
def main():
    getMetrics("Spambase")
    getMetrics("Poker Hand")
    
main()