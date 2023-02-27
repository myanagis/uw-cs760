import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def open_datapoint_file(filename: str, delim: str, file_has_header: bool = False, file_has_data_point_name: bool = False):
    '''

    :param filename: input filename to parse out
    :param delim: Delimiter from the file
    :param has_header:
    :return: data points, data classes
    '''

    # account for some of the other input vars
    data_x = []
    data_y = []
    header_found = False
    header = []
    if file_has_data_point_name == True:
        starting_index = 1
    else:
        starting_index = 0

    # open the file
    with open(filename) as f:
        all_data_reader = csv.reader(f, delimiter=delim)

        for point in all_data_reader:
            if point[0] is None:
                continue

            # if this has a header (should be rare), then
            if (file_has_header is True) and (header_found is False):
                header_found = True
                header = point[starting_index:-1]
                continue

            # otherwise, just add the inputs
            tmp_x = point[starting_index:-1]
            tmp_y = point[-1]
            tmp_x = [float(x) for x in tmp_x]
            tmp_y = float(tmp_y)

            data_x.append(tmp_x)
            data_y.append(tmp_y)

    return data_x, data_y, header


# assumes data_y is a boolean (1/0)
def find_nearest_neighbor(k_arr: int, sample_x, data_x, data_y):

    # get all of the distances (use vectorization so this runs sometime this week)
    tmp_distances = cdist([sample_x], data_x, 'euclidean')

    # get the indexes of the low->high
    sorted_indexes = np.argsort(tmp_distances)
    sorted_indexes = sorted_indexes[0]

    # get the counts
    false_count = 0
    true_count = 0

    ret_array = []
    for k in k_arr:
        for i in range(0,k):
            # get the value of the sorted index
            y_index = sorted_indexes[i]
            y_val = data_y[y_index]

            if int(y_val) == 0:
                false_count += 1
            elif int(y_val) == 1:
                true_count += 1

        if false_count > true_count:
            ret_array.append(0)
        else:
            ret_array.append(1)
    return ret_array



def roc_curve():
    x_axis = [0, .25, .5, 1]
    y_axis = [1/3, 2/3, 1, 1]

    plt.plot(x_axis, y_axis, marker="o")
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

def avg_accuracy_knn():
    x_axis = [1, 3, 5, 7, 10]
    y_axis = [0.8328, .8414, .8412, .8456, .8548]

    plt.plot(x_axis, y_axis, marker="o")
    plt.title('kNN 5-Fold Cross Validation')
    plt.xlabel('k')
    plt.ylabel('Average Accuracy')
    plt.show()

def d2z():
    # get the datapoints from D2z
    data_x, data_y, _ = open_datapoint_file("hw3/D2z.txt", " ")

    # do 1-NN
    linspace = np.linspace(-2, 2, 41)
    test_pt_true_x1 = []
    test_pt_true_x2 = []
    test_pt_false_x1 = []
    test_pt_false_x2 = []
    for tmp_x in linspace:
        for tmp_y in linspace:

            # find the 1-NN
            nearest_y_arr = find_nearest_neighbor([3], [tmp_x, tmp_y], data_x, data_y)
            nearest_y = nearest_y_arr[0]

            # add to a graph strx
            if nearest_y == 1:
                test_pt_true_x1.append(tmp_x)
                test_pt_true_x2.append(tmp_y)
            elif nearest_y == 0:
                test_pt_false_x1.append(tmp_x)
                test_pt_false_x2.append(tmp_y)
            # ax.scatter(false_x1, false_x2, c="r", marker=".", label="y=0")

    # #######
    # partition the data_x and data_y
    false_x1 = []
    false_x2 = []
    true_x1 = []
    true_x2 = []
    for i in range(len(data_y)):
        if data_y[i] == 1:
            true_x1.append(data_x[i][0])
            true_x2.append(data_x[i][1])
        elif data_y[i] == 0:
            false_x1.append(data_x[i][0])
            false_x2.append(data_x[i][1])

    # begin the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the scatter plots
    # https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot

    # 1NN test points
    ax.scatter(test_pt_true_x1, test_pt_true_x2, c="b", s=2, marker=".", label="y=1")
    ax.scatter(test_pt_false_x1, test_pt_false_x2, c="r", s=2, marker=".", label="y=0")

    # input data
    ax.scatter(true_x1, true_x2, c="b", s=20, marker="s", label="y=1")
    ax.scatter(false_x1, false_x2, c="r", s=20, marker="s", label="y=0")

    plt.legend(loc='upper left')
    plt.show()

    # print(data_x)
    # print(data_y)





def run_5_fold_cross_validation(k_arr, data_x, data_y, test_index):
    '''

    :param k:
    :param data_x:
    :param data_y:
    :param test_index: array with 2 values -- [min_index max_index] (inclusive)
    :return:
    '''

    # create the test and train data sets
    training_x = []
    training_y = []
    test_x = []
    test_predicted_y = []
    test_actual_y = []

    # get the min_test_ix and max_test_ix
    min_test_index = test_index[0]
    max_test_index = test_index[1]

    # partition
    # TODO: assert that data_x and data_y have same length. make sure test_index is in range, too.
    for i in range(len(data_y)):

        tmp_x = data_x[i]
        tmp_y = data_y[i]

        # demarcate the test indexes
        if (i >= min_test_index) and (i <= max_test_index):
            test_x.append(tmp_x)
            test_actual_y.append(tmp_y)
        else:
            training_x.append(tmp_x)
            training_y.append(tmp_y)


    # now: use the training data to predict the test data
    accuracy_counter = [0, 0, 0, 0]
    all_accuracy_counters = [accuracy_counter] * (max(k_arr)+1)
    for i in range(len(test_x)):
        tmp_x = test_x[i]
        predicted_y_arr = find_nearest_neighbor(k_arr, tmp_x, training_x, training_y)

        # while we're at it, count TP, FP, TN, FN
        for j in range(len(k_arr)):
            k = k_arr[j]
            predicted_y = predicted_y_arr[j]
            accuracy_counter = all_accuracy_counters[k]
            accuracy_counter = increment_accuracy_counters(accuracy_counter, predicted_y, test_actual_y[i])
            all_accuracy_counters[k] = accuracy_counter


    # calculate all the values
    #print(k, "nearest neighbors, test set indexes:", test_index)

    # accuracy
    print_metrics_from_accuracy_counter(all_accuracy_counters)

def print_metrics_from_accuracy_counter(all_accuracy_counters):
    for i in range(len(all_accuracy_counters)+1):
        counter = all_accuracy_counters[i]
        TP = counter[0]
        FP = counter[1]
        FN = counter[2]
        TN = counter[3]
        if TP <= 0 and FP <=0:
            continue

        print("    TP: ", TP)
        print("    FP: ", FP)
        print("    FN: ", FN)
        print("    TN: ", TN)
        print("    Total in test: ", (TP + TN + FP + FN))
        print("    Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
        if (TP + FP) > 0:
            print("    Precision: ", (TP) / (TP + FP))
        else:
            print("    Precision: ", "0 (TP + FP = 0)")
        if (TP + FN) > 0:
            print("    Recall: ", (TP) / (TP + FN))
        else:
            print("    Recall: ", "0 (TP + FN = 0)")


def increment_accuracy_counters(counter, predicted_result, actual_result):
    # cast both to ints, to be safe
    predicted = int(predicted_result)
    actual = int(actual_result)

    # [TP FP FN TN]

    # True positive
    if predicted == 1 and actual == 1 :
        counter[0] += 1
    # false positive
    elif predicted == 1 and actual == 0:
        counter[1] += 1
    # false negative
    elif predicted == 0 and actual == 1:
        counter[2] += 1
    # true negative
    elif predicted == 0 and actual == 0:
        counter[3] += 1
    return counter


# question 2 - 1NN, plus 5-fold cross validation
def spam_filter_1nn_5foldcrossvalidation():
    data_x, data_y, header = open_datapoint_file("hw3/emails.csv", ",", file_has_header=True, file_has_data_point_name=True)

    #print(data_x[0:5])
    #print(data_y[0:-1])
    #print(header[0:5])

    # great, we now have the 5000 points loaded.
    # implement 5-fold cross-validation
    test_sets = []
    test_sets.append([0, 999]) # note all of the indexes are offset by 1
    test_sets.append([999, 1999])
    test_sets.append([1999, 2999])
    test_sets.append([2999, 3999])
    test_sets.append([3999, 4999])

    # now, just run them tests!
    for test_set in test_sets:
        run_5_fold_cross_validation(1, data_x, data_y, test_set)


def vary_knn():
    data_x, data_y, header = open_datapoint_file("hw3/emails.csv", ",", file_has_header=True, file_has_data_point_name=True)

    #print(data_x[0:5])
    #print(data_y[0:-1])
    #print(header[0:5])

    # great, we now have the 5000 points loaded.
    # implement 5-fold cross-validation
    test_sets = []
    test_sets.append([0, 999]) # note all of the indexes are offset by 1
    test_sets.append([999, 1999])
    test_sets.append([1999, 2999])
    test_sets.append([2999, 3999])
    test_sets.append([3999, 4999])

    # now, just run them tests!
    #TODO: in retrospect, this is pretty inefficient. The "run_5_fold_cross_validation" should just calculate all these different k values as it loops.
    for test_set in test_sets:
        print("Running test set", test_set)
        run_5_fold_cross_validation([1, 3, 5, 7, 10], data_x, data_y, test_set)


def programming_2():
    data_x, data_y, header = open_datapoint_file("hw3/emails.csv", ",", file_has_header=True, file_has_data_point_name=True)

    #print(data_x[0:5])
    #print(data_y[0:-1])
    #print(header[0:5])

    # great, we now have the 5000 points loaded.
    # implement 5-fold cross-validation
    test_sets = []
    test_sets.append([0, 999]) # note all of the indexes are offset by 1
    test_sets.append([999, 1999])
    test_sets.append([1999, 2999])
    test_sets.append([2999, 3999])
    test_sets.append([3999, 4999])

    # now, just run them tests!
    for test_set in test_sets:
        training_x, training_y, test_x, test_y = divvy_up_data(test_set, data_x, data_y)

        for k in [1]:
            print("Test set:",test_set,", k:",k)
            run_knn(k, training_x, training_y, test_x, test_y)


def vary_knn_2():
    data_x, data_y, header = open_datapoint_file("hw3/emails.csv", ",", file_has_header=True, file_has_data_point_name=True)

    #print(data_x[0:5])
    #print(data_y[0:-1])
    #print(header[0:5])

    # great, we now have the 5000 points loaded.
    # implement 5-fold cross-validation
    test_sets = []
    test_sets.append([0, 999]) # note all of the indexes are offset by 1
    test_sets.append([999, 1999])
    test_sets.append([1999, 2999])
    test_sets.append([2999, 3999])
    test_sets.append([3999, 4999])

    # now, just run them tests!
    for test_set in test_sets:
        training_x, training_y, test_x, test_y = divvy_up_data(test_set, data_x, data_y)

        for k in [1, 3, 5, 7, 10]:
            print("Test set:",test_set,", k:",k)
            run_knn(k, training_x, training_y, test_x, test_y)





def run_knn(k, training_x, training_y, test_x, test_y):
    # adapted from https://www.digitalocean.com/community/tutorials/k-nearest-neighbors-knn-in-python
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(training_x, training_y)
    y_predicted = knn.predict(test_x)
    accuracy = accuracy_score(test_y, y_predicted)
    precision = precision_score(test_y, y_predicted)
    recall = recall_score(test_y, y_predicted)
    print("Accuracy: ", accuracy, ", Precision:", precision, ", Recall: ",recall)

def divvy_up_data(test_set_range, data_x, data_y):

    training_x = []
    training_y = []
    test_x = []
    test_actual_y = []

    # get the min_test_ix and max_test_ix
    min_test_index = test_set_range[0]
    max_test_index = test_set_range[1]

    # partition
    # TODO: assert that data_x and data_y have same length. make sure test_index is in range, too.
    for i in range(len(data_y)):

        tmp_x = data_x[i]
        tmp_y = data_y[i]

        # demarcate the test indexes
        if (i >= min_test_index) and (i <= max_test_index):
            test_x.append(tmp_x)
            test_actual_y.append(tmp_y)
        else:
            training_x.append(tmp_x)
            training_y.append(tmp_y)
    return training_x, training_y, test_x, test_actual_y



def implement_logistic_regression():
    data_x, data_y, header = open_datapoint_file("hw3/emails.csv", ",", file_has_header=True,
                                                 file_has_data_point_name=True)

    # great, we now have the 5000 points loaded.
    # implement 5-fold cross-validation
    test_sets = []
    test_sets.append([0, 999])  # note all of the indexes are offset by 1
    test_sets.append([999, 1999])
    test_sets.append([1999, 2999])
    test_sets.append([2999, 3999])
    test_sets.append([3999, 4999])

    # now, just run them tests!
    for test_set in test_sets:
        print("Test set: ", test_set)
        training_x, training_y, test_x, test_y = divvy_up_data(test_set, data_x, data_y)
        run_logistic_regression(training_x, training_y, test_x, test_y)


def run_logistic_regression(training_x, training_y, test_x, test_y):

    lr = LogisticRegression()
    lr.fit(training_x, training_y)
    y_predicted = lr.predict(test_x)

    accuracy = accuracy_score(test_y, y_predicted)
    precision = precision_score(test_y, y_predicted)
    recall = recall_score(test_y, y_predicted)
    print("Accuracy: ", accuracy, ", Precision:", precision, ", Recall: ",recall)



# adapted from https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/
# and https://towardsdatascience.com/logistic-regression-from-scratch-69db4f587e17 (this one is better IMO)
class LogisticRegression:
    # learning rate - 0.01
    # max iter - 10^8??

    def sigma(self, z):
        return 1/( 1 + np.e**(-z))

    def calculate_loss(self, X, y, weights):
        z = np.dot(X, weights)
        sigma_calc = self.sigma(z)
        #print("Sigma: ",sigma_calc)
        log_1 = np.log(sigma_calc)
        #print("Log 1:", log_1)
        #print("y:", y)
        predict_1 = y * log_1
        predict_0 = (1-y) * np.log(1 - self.sigma(z))
        return - sum( predict_1 + [predict_0] ) / len(X)

    def fit(self, X, y, learning_rate = 0.05, iterations = 1000):
        number_of_features = len(X[0])
        number_of_inputs = len(X)

        X = np.array(X)
        y = np.array(y)

        # initialize the weights to something random
        weights = np.random.rand( number_of_features )

        # theta^t+1  =  theta^t - (learning rate) (gradient Loss fn = x(sigma - y) )
        for epoch_number in range(iterations):
            y_hat = self.sigma(np.dot(weights,X.T))
            weights -= learning_rate * np.dot(X.T, y_hat - y) / number_of_inputs
            #loss = self.calculate_loss(X, y, weights)
            #print("Epoch: ", epoch_number)

        self.weights = weights
        #self.loss = loss

    def predict(self, X):
        z = np.dot(X, self.weights)
        ret_array = []
        for i in self.sigma(z):
            if i > 0.5:
                ret_array.append(1)
            else:
                ret_array.append(0)
        return ret_array

    def predict_proba(self, X):
        z = np.dot(X, self.weights)
        return self.sigma(z)

def problem_5_roc_curve():
    data_x, data_y, header = open_datapoint_file("hw3/emails.csv", ",", file_has_header=True,
                                                 file_has_data_point_name=True)

    # single training/test split
    test_set = [0, 3999]

    # divvy up test set
    training_x, training_y, test_x, test_y = divvy_up_data(test_set, data_x, data_y)

    # sklearn: assisted by https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

    # run KNN (k=5) and get ROC curve
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(training_x, training_y)
    y_prob_knn = knn.predict_proba(test_x)

    knn_fpr, knn_tpr, _ = sklearn.metrics.roc_curve(test_y, y_prob_knn[:,1])
    plt.plot(knn_fpr, knn_tpr, label="KNN, AUC = %0.2f" % sklearn.metrics.auc(knn_fpr, knn_tpr), color="blue")


    # run logistic regression and get ROC curve
    lr = LogisticRegression()
    lr.fit(training_x, training_y)
    y_predicted_lr = lr.predict(test_x)
    y_prob_lr = lr.predict_proba(test_x)

    lr_fpr, lr_tpr, _ = sklearn.metrics.roc_curve(test_y, y_prob_lr)
    plt.plot(lr_fpr, lr_tpr, label="Logistic Regression, AUC = %0.2f" % sklearn.metrics.auc(lr_fpr, lr_tpr) , c="orange")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # print a ROC curve
    #roc_curve()


    # programming question #1
    #d2z()

    # programming q #2
    #programming_2()

    # logistic
    #implement_logistic_regression()

    # print 5-fold cross validation kNN (question 4)
    #vary_knn_2()
    #avg_accuracy_knn()

    # problem 5 - ROC
    problem_5_roc_curve()
