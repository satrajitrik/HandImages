import numpy as np

class DecisionTreeClassifier:

    #Initialization
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    #fit transform
    def fit(self, X, y):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)





    #predict single data
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    #predict labels
    def predict(self, X):
        """Predict class for X."""
        return [self._predict(inputs) for inputs in X]






    #gini mpurity
    def _gini(self, y):
        """Compute Gini impurity of a non-empty node.
        Gini impurity is defined as 1 - Σ p^2 over all classes, with p the frequency of a
        class within the node.
        """
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def bestsplit(self, X, y):
        # finds best split by computing information gain

        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)] #class count

        best_gini = self._gini(y)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):

            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1


                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                # The Gini impurity of a split is the weighted average of the Gini impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                # continues when same point
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr



    def _grow_tree(self, X, y, depth=0,l=True):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
        if(depth == 0 and l == True):
            print("Root Node")
        elif l :
            print("Left Node")
        else :
            print("right Node")
        print("num of sample per class : " , node.num_samples_per_class)
        print("predicted_class : " , node.predicted_class)

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self.bestsplit(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                print("selected feature : ", node.feature_index)
                print("Threshold value for selected feature : ", node.threshold)
                print("\n")
                node.left = self._grow_tree(X_left, y_left, depth + 1,True)
                node.right = self._grow_tree(X_right, y_right, depth + 1,False)

        return node






class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None



