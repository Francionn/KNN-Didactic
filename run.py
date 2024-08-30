from machine import Sklearn

iris_data = [
    [5.1, 3.5, 1.4, 0.2, 0],
    [4.9, 3.0, 1.4, 0.2, 0],
    [4.7, 3.2, 1.3, 0.2, 0],
    #[16.3, 13.3, 16.0, 12.5, 10], # add to accuray = 0.5
    #[16.3, 13.3, 16.0, 12.5, 10], #
    #[16.3, 13.3, 16.0, 12.5, 10], #add mote two to accuray = 0.0
    [6.3, 3.3, 6.0, 2.5, 2],
    [5.8, 2.7, 5.1, 1.9, 2]
]

X = [row[:-1] for row in iris_data]
y = [row[-1] for row in iris_data]


machine = Sklearn()
X_train, X_test, y_train, y_test = machine.train_test_split(X,y,8)

distances = machine.fit(X_train, y_train, X_test)

k_neighbors_result = machine.k_neighbors_classifier(distances, 3)

predict = machine.predict(k_neighbors_result)

accuracy = machine.accuracy_score(y_test, predict)

print(accuracy)
