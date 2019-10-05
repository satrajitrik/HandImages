import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


class LatentSymanticsType(object):
    def __init__(self, choice):
        self.symantics_type = self._symantics_type(choice)

    def _symantics_type(self, choice):
        if choice == 1:
            return "pca"
        elif choice == 2:
            return "svd"
        elif choice == 3:
            return "nmf"
        else:
            return "lda"


class LatentSymantics(object):
    def __init__(self, x, k, choice):
        self.x = x
        self.k = k
        self.latent_symantics = self._latent_symantics(choice)

    """
    	Based on choice, call respective class methods.
    	Returns Features with reduced dimensions.
    """

    def _latent_symantics(self, choice):
        if choice == 1:
            return self.pca()
        elif choice == 2:
            return self.svd()
        elif choice == 3:
            return self.nmf()
        else:
            return self.lda()

    """
    	# TODO PCA
    """

    def pca(self):
        pca = PCA(n_components=self.k)
        return pca.fit_transform(self.x)

    """
    	# TODO SVD
    """

    def svd(self):
        svd = TruncatedSVD(n_components=self.k)
        return svd.fit_transform(self.x)

    """
    	# TODO NMF
    """

    def nmf(self):
        nmf = NMF(n_components=self.k)
        return nmf.fit_transform(self.x)

    """
    	LDA Dimensionality Reduction
    """

    def lda(self):
        lda = LatentDirichletAllocation(n_components=self.k)
        return lda.fit_transform(self.x)
