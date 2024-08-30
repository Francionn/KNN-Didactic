import random
import math

from collections import Counter

class Sklearn:

    def train_test_split(self, X, y, test_size):
        data = list(zip(X, y))
        random.seed(42)
        random.shuffle(data)
        X, y = zip(*data)

        split_index = int((test_size/10)* len(X))

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test

    def fit(self, X_train, y_train, X_test):
        
        for test_point in X_test:
             distances =[(self.__euclidean_distance(test_point, train_point), label) 
                        for train_point, label in zip(X_train, y_train)]
        distances.sort(key=lambda x: x[0])
        return distances
    
    def k_neighbors_classifier(self, distances, k ): 
        
        k_nearest_labels = [label for _, label in distances[:k]]
        return k_nearest_labels
    
    def predict(self, k_nearest_labels):
        most_commom = []
        most_commom.append(Counter(k_nearest_labels).most_common(1)[0][0])
        return most_commom

    def accuracy_score(self, y_true, y_predict):
        
        correct = sum(true_y == pred_y for true_y, pred_y in zip(y_true, y_predict))
        return correct / len(y_true)

    def __euclidean_distance(self, first_point, second_point):
        return math.sqrt(sum((x-y)**2 for x , y in zip(first_point, second_point)))
         

    



    