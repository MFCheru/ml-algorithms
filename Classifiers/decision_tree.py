import numpy as np
from classifier import Classifier


def _category_counter(y_train):

    categories_counter = dict() 
    for category in set(y_train):
        categories_counter[category] = 0
    
    for y_label in y_train:
        categories_counter[y_label] += 1

    return categories_counter


class Leaf:

    def __init__(self, y_train):
        self.counter = _category_counter(y_train)

    def get_counter(self):
        return self.counter

class Decision_Node:
    
    def __init__(self, split_criterion, left_sub_tree, right_sub_tree):
        self.split_criterion = split_criterion
        self.left_sub_tree = left_sub_tree
        self.right_sub_tree = right_sub_tree


    def get_leaf(self, x_sample):
        
        current_node = self

        while(type(current_node) != Leaf):
                if (current_node.split_criterion.satisfy(x_sample)):
                    current_node = current_node.left_sub_tree
                else:
                    current_node = current_node.right_sub_tree

        return  current_node


def _remaining_height(height):

    if (height == None):
        return None
    else:
        return height-1


def _build_decision_tree(X_train, y_train, criterion='', height=3):

    gain, split_criterion = _find_best_split(X_train, y_train, criterion)

    # Checks if theres no gain or if theres no remaining height
    if gain == 0 or height <= 0:
        return Leaf(y_train)
    else: 

        # Updates remainig height
        height = _remaining_height(height)
        
        # Separates current instances according to the previosly found attribute
        X_satisfy , Y_satisfy, X_no_satisfy, Y_no_satisfy = _split_according_to(split_criterion, X_train, y_train)

        # Continues construction recursvely with remaining instances.
        left_sub_tree = _build_decision_tree(X_satisfy, Y_satisfy, criterion, height)
        right_sub_tree   = _build_decision_tree(X_no_satisfy, Y_no_satisfy, criterion, height)

        # Connects everithing in a decision node.
        return Decision_Node(split_criterion, left_sub_tree, right_sub_tree)


def _gini(y_train):

    total = len(y_train)
    if (total == 0):
        return 1
    
    category_counter = _category_counter(y_train)
    gini_impurity = 1

    for category in set(y_train):
        gini_impurity = gini_impurity - np.square(category_counter[category] / total)

    return gini_impurity


def _gini_gain(X_train, Y_first_group, Y_second_group):

    total = len(X_train)

    initial_gini = _gini(Y_first_group + Y_second_group)
    gini_first_g = _gini(Y_first_group)
    gini_second_g = _gini(Y_second_group)

    weighted_average = ((len(Y_first_group)/total) * gini_first_g) + ((len(Y_second_group)/total) * gini_second_g)

    return (initial_gini - weighted_average)


def _entropy(y_train):

    total    = len(y_train)
    if (total == 0):
        return 1

    category_counter = _category_counter(y_train)
    
    entropy = 0
    for category in set(y_train) :
        prob = category_counter[category] / total
        if (prob != 0):
            entropy = entropy + (-prob * np.log2(prob))
        
    return entropy


def _entropy_gain(X_train, Y_first_group, Y_second_group):

    total = len(X_train)
    
    initial_entropy = _entropy(Y_first_group + Y_second_group)
    entropy_first_g = _entropy(Y_first_group)
    entropy_second_g = _entropy(Y_second_group)
    
    weighted_average = ((len(Y_first_group)/total) * entropy_first_g ) + ((len(Y_second_group)/total) * entropy_second_g)

    return (initial_entropy - weighted_average)


def _calculate_gain(X_train, Y_first_group, Y_second_group, criterion):

    if (criterion == 'gini'):
        return _gini_gain(X_train, Y_first_group, Y_second_group)
    elif (criterion == 'entropy'):
        return _entropy_gain(X_train, Y_first_group, Y_second_group)
    else:
        raise ValueError("Bad criterion configured")


def _find_best_split(X_train, y_train, criterion):
    max_gain   = 0
    best_split_criterion = None
    n_features = len(X_train[0])

    for feature in range(n_features):
        for x_sample in X_train:

            feature_value = x_sample[feature]
            split_criterion = SplitCriterion(feature, feature_value)
            _, Y_satisfy, _, Y_no_satisfy = _split_according_to(split_criterion, X_train, y_train)
            gain = _calculate_gain(X_train, Y_satisfy, Y_no_satisfy, criterion)

            if gain > max_gain:
                max_gain = gain
                best_split_criterion = split_criterion
            
    return max_gain, best_split_criterion


def _split_according_to(split_criterion, X_train, y_train):
    
    X_satisfy , Y_satisfy, X_no_satisfy, Y_no_satisfy = [], [], [], []
    
    for x_sample, y_sample in zip(X_train, y_train):
        
        if split_criterion.satisfy(x_sample):
            X_satisfy.append(x_sample)
            Y_satisfy.append(y_sample)
        else:
            X_no_satisfy.append(x_sample)
            Y_no_satisfy.append(y_sample)

    return X_satisfy , Y_satisfy, X_no_satisfy, Y_no_satisfy


class SplitCriterion():

    def __init__(self, feature, feature_value):
        self.feature = feature
        self.value = feature_value
    
    def satisfy(self, x_sample):
        return x_sample[self.feature] >= self.value


class DecisionTreeClassifier(Classifier):

    """
    A decision tree classifier.

    Parameters
    ----------
    max_depth : int or None, optional (default=3)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    """

    def __init__(self, max_depth=3, criterion='gini'):
        self.tree     = None
        self.max_depth = max_depth
        self.criterion = criterion
        self.categories = None
    

    def fit(self, X_train, y_train):

        if len(X_train != y_train):
            raise ValueError("Size of X_train and y_train must be equal")

        self.categories = set(y_train)
        self.tree = _build_decision_tree(X_train, y_train, self.criterion, self.max_depth)


    def predict(self, X_test):

        return super().predict(X_test)


    def predict_proba(self, X_test):

        return super().predict_proba(X_test)


    def _predict_sample_proba(self, x_sample):

        if self.tree == None:
            raise ValueError("Model must be fitted previously")

        if type(self.tree) != Leaf:
            leaf = self.tree.get_leaf(x_sample)
        else:
            leaf = self.tree

        total = 0
        probabilities = dict()
        for category in self.categories:

            if category in leaf.get_counter().keys():
                votes = leaf.get_counter()[category]
                probabilities[category] = votes
                total += votes
            else:
                probabilities[category] = 0

        for category in leaf.get_counter().keys():
            leaf.get_counter()[category] /= total

        return probabilities