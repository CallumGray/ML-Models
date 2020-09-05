from Toy import ToyData
from Sarcos import SarcosData
import numpy as np
import time

#
#
#
#   Read data in
#
#
#

#data = ToyData(25000)
data = SarcosData()

print('Std: ',np.std(data.y_train))

# Calculate RMSE of the algorithm
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#
#
#
#   RANDOM FORESTS
#
#
#


def calculate_entropy(y):

    entropy = 0

    for value in np.unique(y):
        proportion = np.count_nonzero(y == value) / len(y)
        entropy += proportion * np.log2(proportion)

    return -entropy


def find_split(x, y):
    """Given a dataset and its target values, this finds the optimal combination
    of feature and split point that gives the maximum information gain."""

    # Need the starting entropy so we can measure improvement...
    start_entropy = calculate_entropy(y)

    # Best thus far, initialised to a dud that will be replaced immediately...
    best = {'infogain': -np.inf}

    # Randomly allocate the splits to be traversed (without replacement)
    feature_total = x.shape[1]
    feature_subset_count = int(np.sqrt(feature_total))
    feature_subset = np.random.permutation(feature_total)[:feature_subset_count]

    # Loop every possible split of every feature...
    for feature_index in feature_subset:
        for split in np.unique(x[:, feature_index]):

            left_indices = []
            right_indices = []

            # Get index of rows where x[row_index,feature_index] <= split
            for row_index,row in enumerate(x):
                left_indices.append(row_index) if x[row_index,feature_index] <= split else right_indices.append(row_index)

            left_ys = y[left_indices]
            right_ys = y[right_indices]

            nleft = len(left_ys)
            nright = len(right_ys)
            ntotal = nleft + nright
            infogain = start_entropy - (nleft / ntotal) * calculate_entropy(left_ys) - (
                        nright / ntotal) * calculate_entropy(right_ys)

            if infogain > best['infogain']:
                best = {'feature': feature_index,
                        'split': split,
                        'infogain': infogain,
                        'left_indices': left_indices,
                        'right_indices': right_indices}
    return best


def build_tree(x, y, max_depth=np.inf,min_leaf_samples=5):
    # Check if either of the stopping conditions have been reached. If so generate a leaf node...
    if max_depth == 1 or len(y) <= min_leaf_samples:
        # Generate a leaf node...
        classification = np.mean(y)

        return {'leaf': True, 'class': classification}

    else:
        #split the data
        move = find_split(x, y)

        left = build_tree(x[move['left_indices'], :], y[move['left_indices']], max_depth - 1)
        right = build_tree(x[move['right_indices'], :], y[move['right_indices']], max_depth - 1)

        return {'leaf': False,
                'feature': move['feature'],
                'split': move['split'],
                'infogain': move['infogain'],
                'left': left,
                'right': right}


def predict_one(tree, sample):
    """Does the prediction for a single data point"""
    if tree['leaf']:
        return tree['class']

    else:
        if sample[tree['feature']] <= tree['split']:
            return predict_one(tree['left'], sample)
        else:
            return predict_one(tree['right'], sample)


def predict(tree, samples):
    """Predicts class for every entry of a data matrix."""
    ret = np.empty(samples.shape[0], dtype=float)
    ret.fill(-1)
    indices = np.arange(samples.shape[0])

    def tranverse(node, indices):
        nonlocal samples
        nonlocal ret

        if node['leaf']:
            ret[indices] = node['class']

        else:
            going_left = samples[indices, node['feature']] <= node['split']
            left_indices = indices[going_left]
            right_indices = indices[np.logical_not(going_left)]

            if left_indices.shape[0] > 0:
                tranverse(node['left'], left_indices)

            if right_indices.shape[0] > 0:
                tranverse(node['right'], right_indices)

    tranverse(tree, indices)
    return ret




def bootstrapped(size):
    train_size = len(data.x_train)
    # select with replacement
    bootstrap_index = np.random.choice(np.arange(train_size),size,True)
    xstrap,ystrap = data.x_train[bootstrap_index],data.y_train[bootstrap_index]
    return xstrap,ystrap


indent_str = '   '


def print_tree(tree, indent = 0):

    if tree['leaf']:
        print(indent_str*indent,"Predict: ",tree['class'])
    else:
        print(indent_str*indent,'[ftr ',tree['feature'],' <= ',tree['split'],']')
        print_tree(tree['left'],indent+1)
        print(indent_str * indent, '[ftr ', tree['feature'], ' > ', tree['split'],']')
        print_tree(tree['right'],indent+1)

def test_quality(forest,x_test,y_test):

    test_predictions = []

    for tree in forest:
        test_predictions.append(predict(tree, x_test))

    final_test_predictions = np.mean(test_predictions, axis=0)
    test_error = rmse(final_test_predictions, y_test)

    return test_error

def optimise_hyperparameters():

    training_time = time.time()

    TREES = [10, 20, 30]
    DEPTH = [2, 4, np.inf]
    MIN_LEAF_SAMPLES = [2, 4, 6, 8]

    best_trees = 10
    best_depth = 2
    best_leaf_samples = 2
    best_error = np.inf

    for t in TREES:
        for d in DEPTH:
            for s in MIN_LEAF_SAMPLES:
                rf = create_forest(t,d,s)

                rf_v_err = test_quality(rf,data.x_validation,data.y_validation)

                if rf_v_err < best_error:
                    best_trees = t
                    best_depth = d
                    best_leaf_samples = s
                    best_error = rf_v_err

    training_time = time.time() - training_time
    print('Tuning Time on Length', len(data.x_train), ':', training_time)
    print('Best number of trees:', best_trees)
    print('Best min samples at leaves:', best_leaf_samples)
    print('Best tree depth:', best_depth)

    build_time = time.time()
    final_rf = create_forest(best_trees,best_depth,best_leaf_samples)
    build_time = time.time() - build_time
    print('Build Time on Length:', len(data.x_train), ':', build_time)

    predict_all_time = time.time()
    rf_test_err = test_quality(final_rf,data.x_test,data.y_test)
    predict_all_time = time.time() - predict_all_time
    print('Test Time on Length', len(data.x_test), ':', predict_all_time)

    rf_train_err = test_quality(final_rf,data.x_train,data.y_train)
    print('Train Standard Deviation: ', np.std(data.y_train))
    print('Test Standard Deviation: ', np.std(data.y_test))
    print('RANDOM FOREST TRAIN ERROR: ', rf_train_err)
    print('RANDOM FOREST TEST ERROR: ', rf_test_err)

def create_forest(trees,depth,min_leaf_samples):
    forest = []
    BOOTSTRAP_SIZE = int(np.sqrt(len(data.x_train)))

    for i in range(trees):
        x,y = bootstrapped(BOOTSTRAP_SIZE)
        tree = build_tree(x,y,depth,min_leaf_samples)
        forest.append(tree)
        print(i+1,'/',trees)
        #print_tree(tree)

    return forest

#test_forest = create_forest(50,np.inf,3)
#error = test_quality(test_forest,data.x_test,data.y_test)
#print(error)

optimise_hyperparameters()
