import numpy as np
from collections import OrderedDict
import cvxopt
from cvxopt import matrix,solvers



class LimitedSizeDict(OrderedDict):

    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class Base_binary_classification(object):

    def __init__(self, kernel, C=1.0, support_vector_tol=0.0):
        self.kernel = kernel
        self.C = C
        self.support_vectors_ = []
        self.dual_coef_ = []
        self.intercept_ = 0.0
        self.support_vector_tol = support_vector_tol

    def fit(self, X, y):
        self.shape_fit_ = X.shape

        res_compute_weights = self._compute_weights(X, y)
        intercept_already_computed = type(res_compute_weights) == tuple
        if intercept_already_computed:
            lagrange_multipliers, intercept = res_compute_weights
            self.intercept_ = intercept
        else:
            lagrange_multipliers = res_compute_weights



        support_vector_indices = lagrange_multipliers > self.support_vector_tol

        self.support_ = (support_vector_indices * range(self.shape_fit_[0])).nonzero()[0]
        if support_vector_indices[0]:
            self.support_ = np.insert(self.support_, 0, 0)

        self.dual_coef_ = lagrange_multipliers[support_vector_indices] * y[support_vector_indices]
        self.support_vectors_ = X[support_vector_indices]
        self.n_support_ = np.array([sum(y[support_vector_indices] == -1),
                                    sum(y[support_vector_indices] == 1)])

        if not intercept_already_computed:
            self.intercept_ = np.mean(y[support_vector_indices] - self.predict(self.support_vectors_))

    def _compute_weights(self, X, y):
        raise NotImplementedError()

    def _compute_kernel_support_vectors(self, X):
        res = np.zeros((X.shape[0], self.support_vectors_.shape[0]))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(self.support_vectors_):
                res[i, j] = self.kernel(x_i, x_j)
        return res

    def _predict_proba(self, X, kernel_support_vectors=None):
        if kernel_support_vectors is None:
            kernel_support_vectors = self._compute_kernel_support_vectors(X)
        prod = np.multiply(kernel_support_vectors, self.dual_coef_)
        prediction = self.intercept_ + np.sum(prod, 1)  # , keepdims=True)
        print(prediction)
        return prediction

    def predict_proba(self, X):
        return self._predict_proba(X)

    def predict(self, X):
        return np.sign(self._predict_proba(X))

    def predict_value(self, X):
        n_samples = X.shape[0]
        prediction = np.zeros(n_samples)
        for i, x in enumerate(X):
            result = self.intercept_
            for z_i, x_i in zip(self.dual_coef_,
                                self.support_vectors_):
                result += z_i * self.kernel(x_i, x)
            prediction[i] = result
        return prediction

    def score(self, X, y):

        prediction = self.predict(X)
        scores = prediction == y
        return sum(scores) / len(scores)


class binary_classification_qp(Base_binary_classification):

    def __init__(self, kernel, C=1.0):
        super().__init__(kernel, C=C)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self.kernel(x_i, x_j)
        return K

    def _compute_weights(self, x, y):

        m = len(y)
        print (x.shape)
        print(y.shape)
        K = self._gram_matrix(x)
        P = K * np.outer(y, y)
        P, q = matrix(P, tc='d'), matrix(-np.ones((m, 1)), tc='d')
        G = matrix(np.r_[-np.eye(m), np.eye(m)], tc='d')
        h = matrix(np.r_[np.zeros((m, 1)), np.zeros((m, 1)) + self.C], tc='d')
        A, b = matrix(y.reshape((1, -1)), tc='d'), matrix([0.0])
        solution = solvers.qp(P, q, G, h, A, b)
        if solution['status'] == 'unknown':
            print
            'Not PSD!'
            exit(2)
        else:
            self.alphas = np.array(solution['x']).squeeze()
        return np.ravel(solution['x'])

class binary_classification_smo(Base_binary_classification):

    def __init__(self, kernel, C=1.0, max_iter=1000, cache_size=200, tol=0.001):
        super().__init__(kernel, C)
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.tol = tol
        self._reset_cache_kernels()

    def _compute_kernel_matrix_row(self, X, index):
        if self.cache_size != 0 and index in self._cache_kernels:
            return self._cache_kernels[index]
        row = np.zeros(X.shape[0])
        x_i = X[index, :]
        for j, x_j in enumerate(X):
            row[j] = self.kernel(x_i, x_j)
        self._cache_kernels[index] = row
        return row

    def _compute_kernel_matrix_diag(self, X):
        n_samples, n_features = X.shape
        diag = np.zeros(n_samples)
        for j, x_j in enumerate(X):
            diag[j] = self.kernel(x_j, x_j)
        return diag

    def _reset_cache_kernels(self):
        self._cache_kernels = LimitedSizeDict(size_limit=self.cache_size)

    def _compute_intercept(self, alpha, yg):
        indices = (alpha < self.C) * (alpha > 0)
        if len(indices) > 0:
            return np.mean(yg[indices])
        else:
            print('INTERCEPT COMPUTATION ISSUE')
            return 0.0

    def _compute_weights(self, X, y):
        iteration = 0
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        g = np.ones(n_samples)
        self._reset_cache_kernels()
        while True:

            yg = g * y
            indices_y_pos = (y == 1)
            indices_y_neg = (np.ones(n_samples) - indices_y_pos).astype(bool)  # (y == -1)
            indices_alpha_big = (alpha >= self.C)
            indices_alpha_neg = (alpha <= 0)

            indices_violate_Bi_1 = indices_y_pos * indices_alpha_big
            indices_violate_Bi_2 = indices_y_neg * indices_alpha_neg
            indices_violate_Bi = indices_violate_Bi_1 + indices_violate_Bi_2
            yg_i = yg.copy()
            yg_i[indices_violate_Bi] = float('-inf')

            indices_violate_Ai_1 = indices_y_pos * indices_alpha_neg
            indices_violate_Ai_2 = indices_y_neg * indices_alpha_big
            indices_violate_Ai = indices_violate_Ai_1 + indices_violate_Ai_2
            yg_j = yg.copy()
            yg_j[indices_violate_Ai] = float('+inf')

            i = np.argmax(yg_i)
            Ki = self._compute_kernel_matrix_row(X, i)
            Kii = Ki[i]

            j = np.argmin(yg_j)
            Kj = self._compute_kernel_matrix_row(X, j)

            stop_criterion = yg_i[i] - yg_j[j] < self.tol
            if stop_criterion or (iteration >= self.max_iter and self.max_iter != -1):
                break

            min_1 = (y[i] == 1) * self.C - y[i] * alpha[i]
            min_2 = y[j] * alpha[j] + (y[j] == -1) * self.C
            min_3 = (yg_i[i] - yg_j[j]) / (Kii + Kj[j] - 2 * Ki[j])
            lambda_param = np.min([min_1, min_2, min_3])

            # update gradient
            g = g + lambda_param * y * (Kj - Ki)
            alpha[i] = alpha[i] + y[i] * lambda_param
            alpha[j] = alpha[j] - y[j] * lambda_param

            iteration += 1
        # compute intercept
        intercept = self._compute_intercept(alpha, yg)

        print('{} iterations for gradient ascent'.format(iteration))
        self._reset_cache_kernels()
        return alpha, intercept