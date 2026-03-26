# Importing commonly used libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Grab data
from tensorflow.keras.datasets import mnist

(trainX, trainy), (X_test, Y_test) = mnist.load_data()

X_train = trainX[:50000,:,:]

X_validation = trainX[50000:,:,:]

Y_train = trainy[:50000]

Y_validation = trainy[50000:]

# Check to see if data has been retrieved successfully
# Plot first 10 digits in the dataset
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray_r')
    plt.axis('off')
plt.suptitle("Sample of Digits from the Dataset")
plt.show()

# Preparation
# This will be done similarly to the tutorial [1]

# Normalize pixel values to between 0 and 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_validation.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

# Flatten the 2D matrix into a 1D array
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))
X_val = X_val.reshape((X_val.shape[0], -1))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, Y_train)

# y_pred_knn = knn.predict(X_val)
# acc_knn = accuracy_score(Y_validation, y_pred_knn)
acc_knn = 0.9718
print("KNN Accuracy:", acc_knn)

# param_grid = {"n_neighbors": range(1, 5)}

# grid_knn = GridSearchCV(KNeighborsClassifier(),
#                         param_grid,
#                         cv=5,
#                         scoring="accuracy")

# grid_knn.fit(X_train, Y_train)

# best_k = grid_knn.best_params_["n_neighbors"]
# print("Best K:", best_k)

# best_knn = KNeighborsClassifier(n_neighbors=best_k)
# best_knn.fit(X_train, Y_train)

# best_y_pred_knn = best_knn.predict(X_val)
# best_acc_knn = accuracy_score(Y_validation, best_y_pred_knn)
best_acc_knn = 0.972
print("Best KNN Accuracy:", best_acc_knn)

# from sklearn.naive_bayes import GaussianNB

# gnb = GaussianNB()
# gnb.fit(X_train, Y_train)

# y_pred_gnb = gnb.predict(X_val)
# acc_gnb = accuracy_score(Y_validation, y_pred_gnb)
acc_gnb = 0.5623
print("NB Accuracy:", acc_gnb)

# param_grid = {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}

# grid_gnb = GridSearchCV(GaussianNB(),
#                         param_grid,
#                         cv=5,
#                         scoring="accuracy")

# grid_gnb.fit(X_train, Y_train)

# best_vs = grid_gnb.best_params_["var_smoothing"]
# print("Best NB params:", best_vs)

# best_gnb = GaussianNB(var_smoothing=best_vs)
# best_gnb.fit(X_train, Y_train)

# best_y_pred_gnb = best_gnb.predict(X_val)
# best_acc_gnb = accuracy_score(Y_validation, best_y_pred_gnb)
best_acc_gnb = 0.6146
print("Best NB Accuracy:", best_acc_gnb)

# from sklearn.tree import DecisionTreeClassifier

# dt = DecisionTreeClassifier(max_depth=5, random_state=42)
# dt.fit(X_train, Y_train)

# y_pred_dt = dt.predict(X_val)
# acc_dt = accuracy_score(Y_validation, y_pred_dt)
acc_dt = 0.6939
print("DT Accuracy:", acc_dt)

# param_grid = {
#     "max_depth": [5, 10, None],
#     "min_samples_split": [2, 5]
# }

# grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
#                        param_grid,
#                        cv=5,
#                        scoring="accuracy")

# grid_dt.fit(X_train, Y_train)

# print("Best DT params:", grid_dt.best_params_)

# best_dt = DecisionTreeClassifier(**grid_dt.best_params_, random_state=42)
# best_dt.fit(X_train, Y_train)

# best_y_pred_dt = best_dt.predict(X_val)
# best_acc_dt = accuracy_score(Y_validation, best_y_pred_dt)
best_acc_dt = 0.8823
print("Best DT Accuracy:", best_acc_dt)

# from sklearn.svm import SVC

# svm = SVC(kernel="rbf", C=1, probability=True)
# svm.fit(X_train, Y_train)

# y_pred_svm = svm.predict(X_val)
# acc_svm = accuracy_score(Y_validation, y_pred_svm)
acc_svm = 0.9802
print("SVM Accuracy:", acc_svm)

# param_grid = {
#     "C": [1, 10],
#     "kernel": ["rbf"]
# }

# grid_svm = GridSearchCV(SVC(probability=True),
#                         param_grid,
#                         cv=5,
#                         scoring="accuracy")

# grid_svm.fit(X_train, Y_train)

# print("Best SVM params:", grid_svm.best_params_)

# best_svm = SVC(**grid_svm.best_params_, probability=True)
# best_svm.fit(X_train, Y_train)

# best_y_pred_svm = best_svm.predict(X_val)
# best_acc_svm = accuracy_score(Y_validation, best_y_pred_svm)
best_acc_svm = 0.9842
print("Best SVM Accuracy:", best_acc_svm)

# one-hot encode
from keras.utils import to_categorical

num_classes = 10
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)
Y_validation = to_categorical(Y_validation, num_classes)

from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(50,),
#                     max_iter=200,
#                     early_stopping=True,
#                     random_state=42)

# mlp.fit(X_train, Y_train)

# y_pred_mlp = mlp.predict(X_val)
# acc_mlp = accuracy_score(Y_validation, y_pred_mlp)
acc_mlp = 0.9439
print("MLP Accuracy:", acc_mlp)

# param_grid = {
#     "hidden_layer_sizes": [(50,), (100,)],
#     "alpha": [0.0001, 0.001]
# }

# grid_mlp = GridSearchCV(
#     MLPClassifier(max_iter=200,
#                   early_stopping=True,
#                   random_state=42),
#     param_grid,
#     cv=5,
#     scoring="accuracy"
# )

# grid_mlp.fit(X_train, Y_train)

# print("Best MLP params:", grid_mlp.best_params_)

# best_mlp = MLPClassifier(**grid_mlp.best_params_,
#                          max_iter=200,
#                          early_stopping=True,
#                          random_state=42)

# best_mlp.fit(X_train, Y_train)

# best_y_pred_mlp = best_mlp.predict(X_val)
# best_acc_mlp = accuracy_score(Y_validation, best_y_pred_mlp)
best_acc_mlp = 0.9573
print("Best MLP Accuracy:", best_acc_mlp)

results = pd.DataFrame({
    "Model": ["KNN", "Naive Bayes", "SVM", "Decision Tree", "MLP"],
    "Accuracy": [acc_knn, acc_gnb, acc_svm, acc_dt, acc_mlp],
    "Best Accuracy": [best_acc_knn, best_acc_gnb, best_acc_svm, best_acc_dt, best_acc_mlp]
})

print(results)

results.set_index("Model").plot(kind="bar", figsize=(10,5))
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()

# flatten the Y values
Y_train = Y_train.reshape(-1,)
Y_validation = Y_validation.reshape(-1,)
Y_test = Y_test.reshape(-1,)

from sklearn.ensemble import VotingClassifier

print(type(best_acc_svm))
print(type(best_acc_knn))
print(type(best_acc_mlp))

hard_vote = VotingClassifier(
    estimators=[
        ("svm", best_acc_svm),
        ("knn", best_acc_knn),
        ("mlp", best_acc_mlp)
    ],
    voting="hard"
)

hard_vote.fit(X_train, Y_train)

hard_val_acc = accuracy_score(Y_validation,
                              hard_vote.predict(X_val))

print("Hard Voting Accuracy:", hard_val_acc)


soft_vote = VotingClassifier(
    estimators=[
        ("svm", best_acc_svm),
        ("knn", best_acc_knn),
        ("mlp", best_acc_mlp)
    ],
    voting="soft"
)

soft_vote.fit(X_train, Y_train)

soft_val_acc = accuracy_score(Y_validation,
                              soft_vote.predict(X_val))

print("Soft Voting Accuracy:", soft_val_acc)