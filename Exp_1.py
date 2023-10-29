# Import datasets, classifiers and performance metrics
from sklearn import  metrics, svm
from sklearn.model_selection import train_test_split
from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]

#1 get data sets
X, y =read_digits()

X_train, X_test, X_dev, y_train, y_test, y_dev= train_test_dev_split(X, y, test_size=0.3, dev_size=0.3)


# data preprocessing
X_train= preprocess_data(X_train)
X_test= preprocess_data(X_test)
X_dev = preprocess_data(X_dev)



#hyper parameter tuning
best_acc_so_for = -1
Best_model=None
for cur_gamma in gamma_ranges:
    for cur_C in C_ranges:
        # print("running for gamma={}) C = {}". format(cur_gamma, cur_C))
        #-train_model with cur_gamma and cur_C
        cur_model = train_model(X_train, y_train, {'gamma': cur_gamma, 'C': cur_C }, model_type="svm")
        # get some prefformance matrix on Dev set
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        # select the hparams that yield the best proformance on DEV
        if cur_accuracy > best_acc_so_for:
            print("new best accuracy: ", cur_accuracy)
            best_acc_so_for = cur_accuracy
            optimal_gamma = cur_gamma 
            optimal_C =  cur_C
            Best_model= cur_model 
print("optimal_gamma: ", optimal_gamma, "C : ", optimal_C)
# model trainning
model = train_model(X_train, y_train, {'gamma': optimal_gamma, 'C': optimal_C }, model_type="svm")


# Predict the value of the digit on the test subset
#predicted = model.predict(X_test)
test_accuracy = predict_and_eval(Best_model, X_test, y_test)
print("test accuracy: ", test_accuracy)