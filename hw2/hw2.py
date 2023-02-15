
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from numpy.polynomial.polynomial import Polynomial

import scipy.interpolate
from sklearn import tree
from scipy.interpolate import lagrange

def determine_candidate_splits(data):
    x1_vals = []
    x2_vals = []

    if data == None:
        return x1_vals, x2_vals

    if len(data) == 0:
        return x1_vals, x2_vals

    for point in data:
        x1 = point[0]
        x2 = point[1]

        if x1 not in x1_vals:
            x1_vals.append(x1)

        if x2 not in x2_vals:
            x2_vals.append(x2)
    return x1_vals, x2_vals


def has_stopping_criteria_been_met(data, x1_info_gain, x2_info_gain):

    # node empty
    if data == None:
        return True
    if len(data) == 0:
        return True

    # all splits have zero gain ratio
    has_non_zero_gain_ratio = False
    stopping_criteria_met = False
    for x1 in x1_info_gain:
        # info gain is [entropy,info_gain,gain_ratio]
        if x1[0] == 0:
            stopping_criteria_met=True
        if x1[2] > 0:
            has_non_zero_gain_ratio=True

    if stopping_criteria_met == True:
        return True

    # TODO this needs to be refactored
    for x2 in x2_info_gain:
        # info gain is [entropy,info_gain,gain_ratio]
        if x2[0] == 0:
            stopping_criteria_met = True
        if x2[2] > 0:
            has_non_zero_gain_ratio = True
    if stopping_criteria_met == True:
        return True
    if has_non_zero_gain_ratio == False:
        return True
    return False

# yea we assume a 2 probabilities total
def log_calculation(probability):
    other_probability = 1-probability
    if probability==0 or probability == 1:
        return 0
    return (- (probability * math.log(probability, 2)) - (other_probability * math.log(other_probability, 2)))

def find_best_split(data, x1_splits, x2_splits):
    # use info gain ratio. break ties arbitrarily
    x1_info_gain = []
    x2_info_gain = []
    info_gain_max_val = 0
    best_split_x_set = 0
    best_split_x_value = 0

    # first calculate entropy
    y_is_1_count=0
    total=0
    if data == None:
        return best_split_x_set,best_split_x_value,x1_info_gain,x2_info_gain

    for point in data:
        if point[2] == "1":
            y_is_1_count += 1
        total += 1
    prob_y_equals_1 = y_is_1_count/total #TODO: catch total=0
    entropy = log_calculation(prob_y_equals_1)

    # loop through x1 splits
    for x1 in x1_splits:
        #print("X1 partition: ", x1)
        greater_yes_count = 0
        less_than_yes_count = 0
        total_count_greater = 0
        total_count_less = 0
        total_total = 0


        for point in data:
            if point[0] >= x1:
                if point[2] == "1":
                    greater_yes_count += 1
                total_count_greater += 1
            else:
                if point[2] == "1":

                    less_than_yes_count += 1
                total_count_less += 1
            total_total += 1

        #print("\tGreater than yes count: ",greater_yes_count)
        #print("\tTOTAL Greater than yes count: ", total_count_greater)
        #print("\tLess than yes count: ", less_than_yes_count)
        #print("\tTOTAL Less than yes count: ", total_count_less)

        total_weighted_info = 0
        if total_count_greater>0:
            total_weighted_info = total_count_greater/total_total * log_calculation(greater_yes_count/total_count_greater)
        if total_count_less>0:
            total_weighted_info = total_weighted_info + (total_count_less/total_total *  log_calculation(less_than_yes_count/total_count_less))

        info_gain = 0
        gain_ratio=0
        info_gain = entropy - total_weighted_info
        entropy_in_set = log_calculation(total_count_greater/total_total)
        if entropy_in_set > 0:
            gain_ratio = info_gain/entropy_in_set
        #print("\tEntropy: ", entropy)
        #print("\tInfo gain: ", info_gain)
        #print("\tInfo gain ratio: ", gain_ratio)
        x1_info_gain.append([entropy,info_gain,gain_ratio])

        if gain_ratio>info_gain_max_val:
            info_gain_max_val=gain_ratio
            best_split_x_set = 1
            best_split_x_value = x1


    for x2 in x2_splits:
        #print("X2 partition: ", x2)
        greater_yes_count = 0
        less_than_yes_count = 0
        total_count_greater = 0
        total_count_less = 0
        total_total = 0


        for point in data:
            if point[1] >= x2:
                if point[2] == "1":
                    greater_yes_count += 1
                total_count_greater += 1
            else:
                if point[2] == "1":

                    less_than_yes_count += 1
                total_count_less += 1
            total_total += 1

        #print("\tGreater than yes count: ",greater_yes_count)
        #print("\tTOTAL Greater than yes count: ", total_count_greater)
        #rint("\tLess than yes count: ", less_than_yes_count)
        #print("\tTOTAL Less than yes count: ", total_count_less)

        total_weighted_info = 0
        if total_count_greater>0:
            total_weighted_info = total_count_greater/total_total * log_calculation(greater_yes_count/total_count_greater)
        if total_count_less>0:
            total_weighted_info = total_weighted_info + (total_count_less/total_total *  log_calculation(less_than_yes_count/total_count_less))

        info_gain = 0
        gain_ratio=0
        info_gain = entropy - total_weighted_info
        entropy_in_set = log_calculation(total_count_greater/total_total)
        if entropy_in_set > 0:
            gain_ratio = info_gain/entropy_in_set
        #print("\tEntropy: ", entropy)
        #print("\tInfo gain: ", info_gain)
        #print("\tInfo gain ratio: ", gain_ratio)
        x1_info_gain.append([entropy, info_gain, gain_ratio])

        if gain_ratio>info_gain_max_val:
            info_gain_max_val=gain_ratio
            best_split_x_set = 2
            best_split_x_value = x2

    return best_split_x_set,best_split_x_value,x1_info_gain,x2_info_gain

def get_y_value_for_data(data):
    # because we assume this splits perfectly, we need no threshold.
    # just grab first value
    if data == None:
        return ""
    return data[0][2]

#splits et should be 1 or 2
def split_data(data, split_set, split_value, isGreaterThan):
    newdata=[]

    for point in data:
        tmp_val = point[split_set-1]

        if isGreaterThan == True:
            if tmp_val >= split_value:
                newdata.append(point)
        else:
            if tmp_val < split_value:
                newdata.append(point)
    return newdata

# data input is 2D and is continuous (x is element of reals)
# class label is binary (0,1)
# data files are plaintext: x11, x12, y1
def make_subtree(data):
    x1_splits, x2_splits = determine_candidate_splits(data)

    # determine splits
    best_split_x_set, best_split_x_value, x1_info_gain, x2_info_gain = find_best_split(data, x1_splits, x2_splits)

    # LEAF
    if has_stopping_criteria_been_met(data, x1_info_gain, x2_info_gain):
        leaf_class =  get_y_value_for_data(data) # make leaf node N
        node_to_return = ["leaf", leaf_class]

    # INTERNAL NODE
    else:
        # make internal node N

        # for each group K in best_splits:
        # make the subtree of the subset of training data

        # we have the best split. partition based on geq / lessthan
        greater_subset = split_data(data, best_split_x_set, best_split_x_value, True)
        #print("Greater subset: ",greater_subset)
        less_than_subset = split_data(data, best_split_x_set, best_split_x_value, False)
        node_to_return = ["split", best_split_x_set, best_split_x_value, make_subtree(greater_subset), make_subtree(less_than_subset)]

    return node_to_return

# problem 3
def druns():
    data = []
    with open('Druns.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point[0] is not None:
                data.append(point)

    #print(data)
    x1_splits, x2_splits = determine_candidate_splits(data)
    print("X1 splits: ", x1_splits)
    print("X2 splits: ", x2_splits)

    find_best_split(data, x1_splits, x2_splits)

def test_all_zero_info_gain():
    data = []
    with open('greedy.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            data.append(point)

    #print(data)
    x1_splits, x2_splits = determine_candidate_splits(data)
    print("X1 splits: ", x1_splits)
    print("X2 splits: ", x2_splits)

    find_best_split(data, x1_splits, x2_splits)

def test_tree():
    data = []
    with open('test.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point:
                data.append(point)

    tree = make_subtree(data)
    print(tree)

# problem 4
def d3_leaves():
    data = []
    with open('D3leaves.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point:
                data.append(point)

    tree = make_subtree(data)
    show_tree(tree, 0)

def show_tree(tree,level):
    type = tree[0]


    if type == "leaf":
        print("-"*(level*2),"-- Y=", tree[1])
    elif type == "split":
        x_set = tree[1]
        x_val = tree[2]
        greaterthan = tree[3]
        lessthan = tree[4]

        print("-"*(level*2), "IF X", x_set,"\geq",x_val,":")
        show_tree(greaterthan, level + 1)
        print("-"*(level*2), "ELSE:")
        show_tree(lessthan, level + 1)

# problem 5
def or_is_it_d1():
    data = []
    with open('D1.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point:
                data.append(point)

    tree = make_subtree(data)
    show_tree(tree, 0)


def or_is_it_d2():
    data = []
    with open('D2.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point:
                data.append(point)

    tree = make_subtree(data)
    show_tree(tree, 0)

def plot_data():
    data = []
    x1_true = []
    x1_false = []
    x2_true = []
    x2_false = []
    with open('D1.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point:
                #data.append(point)
                if point[2] == "1":
                    x1_true.append(point[0])
                    x2_true.append(point[1])
                    print("TRUE: ", point[0], ", ", point[1])
                else:
                    x1_false.append(point[0])
                    x2_false.append(point[1])

    print("X1 true: ", x1_true)
    print("X2 true: ", x2_true)
    plt.scatter(x1_true,x2_true)
    print("X1 false: ", x1_false)
    print("X2 false: ", x2_false)
    plt.scatter(x1_false, x2_false, c="red")
    #plt.axis('equal')
    #plt.plot(x1_true,x2_true, 'o')
    #plt.grid()
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)

    plt.show()

# problem 7
def learning_curve():
    data = []
    with open('Dbig.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point:
                data.append(point)

    # partition
    # 32, 128, 512, 2048, 8192
    shuffle(data)

    for size in [32, 128, 512, 2048, 8192]:
        data_part = data[0:size-1]
        test_data = data[size:]

        # make tree
        tree = []
        tree = make_subtree(data_part)
        error_rate = calculate_error_rate(tree, test_data)
        print("SIZE: ", size)
        print("ERROR RATE: ", error_rate)


    #tree = make_subtree(data)
    f = open("randomized_data.csv", 'w')
    for point in data:
        if point:
            print(point[0], " ", point[1], " ", point[2])


def calculate_y_from_tree(tree, x1, x2):
    if tree is None:
        return 0
    node_name = tree[0]
    if node_name == "leaf":
        return tree[1]
    elif node_name == "split":
        # ["split", best_split_x_set, best_split_x_value, make_subtree(greater_subset), make_subtree(less_than_subset)]
        x_set = tree[1]
        x_value = tree[2]
        greater_than = tree[3]
        less_than = tree[4]

        if x_set == 1:
            if float(x1) >= float(x_value):
                return calculate_y_from_tree(greater_than,x1,x2)
            else:
                return calculate_y_from_tree(less_than,x1,x2)
        elif x_set == 2:
            if float(x2) >= float(x_value):
                return calculate_y_from_tree(greater_than,x1,x2)
            else:
                return calculate_y_from_tree(less_than,x1,x2)

def calculate_error_rate(tree, data):
    total = 0
    error = 0
    for point in data:
        # evaluate the point, according to the tree
        calculated = calculate_y_from_tree(tree, point[0], point[1])
        actual = point[2]

        total += 1
        if (str(calculated) != str(actual)):
            error +=1

    if total == 0:
        return 0
    return (error/total*100)

# section 3
def use_sci_kit_learn():
    data = []
    with open('Dbig.txt') as f:
        all_data_reader = csv.reader(f, delimiter=" ")
        print(all_data_reader)
        for point in all_data_reader:
            if point:
                data.append(point)

    # partition
    # 32, 128, 512, 2048, 8192
    shuffle(data)

    for size in [32, 128, 512, 2048, 8192]:
        data_part = data[0:size-1]
        test_data = data[size:]

        # make tree, from https://scikit-learn.org/stable/modules/tree.html

        # get decision tree
        all_x, all_y = transform_to_scikit_strx(data_part)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(all_x, all_y)

        # use test set
        test_x, actual_y = transform_to_scikit_strx(test_data)
        predictions = clf.predict(test_x)

        # compare actual_y vs. predictions
        error = 0
        total = 0
        for i in range(len(actual_y)):
            if int(actual_y[i]) != int(predictions[i]):
                error += 1
            total += 1

        if total == 0:
            error_rate = 0
        else:
            error_rate = error/total*100
        print("SIZE: ", size)
        print("ERROR RATE: ", error_rate)

def transform_to_scikit_strx(data):
    all_x = []
    all_y = []
    for point in data:
        x = [point[0], point[1]]
        y = point[2]

        all_x.append(x)
        all_y.append(y)
    return all_x, all_y

#section 4: lagrange
def lagrange_problem():
    start = 0
    end = 5
    count = 10
    gaussian_noise_std_dev = .00001


    # sample
    sample = np.random.uniform(start, end, count)
    sample.sort()
    #print(sample)
    #print(sample[0], ", ", sample[1])


    tmp = []
    for x in sample:
        tmp.append(math.sin(x)) # TODO: there's definitely a better way

    y = np.array(tmp)

    # add gaussian noise
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    noise = np.random.normal(0, gaussian_noise_std_dev, count)
    #print("Noise: ",noise)

    noised_sample = []
    for i in range(count):
        noised_sample.append(sample[i] + noise[i])
    # lagrange
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html
    poly = lagrange(noised_sample, y)

    # calculate polynomial at all values of sample
    predicted = poly(sample)
    #print(predicted)
    # Error
    # calculate RMS error:
    print("RMS error (train data actual vs. lagrange): ", calculate_rms_error(predicted,y))


    # Generate new test data

    # sample
    sample2 = np.random.uniform(start, end, count)
    sample2.sort()
    tmp = []
    for x in sample2:
        tmp.append(math.sin(x))  # TODO: there's definitely a better way
    y2 = np.array(tmp)
    predicted2 = poly(sample2)
    print("RMS error (test data actual vs. lagrange): ", calculate_rms_error(predicted2, y2))



    # add gaussian noise

    # calculate error for training data

    # calculate error for

    x_poly = np.arange(start,end,.002)
    y_poly = poly(x_poly)

    #plt.scatter(sample, y)
    plt.scatter(x_poly,y_poly)
    plt.show()
    return

    #print(poly)

    #plt.plot(x_poly,y_poly)
    #plt.show()
    #plt.scatter(sample, y, label='data')
    #plt.plot(x_new, Polynomial(poly.coef[::-1])(x_new), label='Polynomial')
    #plt.legend()
    #plt.show()

def calculate_rms_error(predicted,actual):
    squared_sum = 0
    for i in range(len(predicted)):
        p = predicted[i]
        a = actual[i]
        squared_sum += ((p - a)**2)
    return math.sqrt(squared_sum)



if __name__ == '__main__':
    #
    #druns()

    # problem

    # problem 4
    #d3_leaves()

    # problem 5
    #or_is_it_d2()

    # problem 6
    #plot_data()

    # problem 7
    #learning_curve()

    # problem 8
    #use_sci_kit_learn()

    # section 4
    lagrange_problem()

