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

    def check_nmf_lda(self):
        if self.symantics_type == "nmf" or self.symantics_type == "lda":
            return True
        return False


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
    	PCA Dimensionality Reduction
    """

    def pca(self):
        if 0 < self.k < min(self.x.shape[0], self.x.shape[1]):
            pca = PCA(n_components=self.k)
        else:
            pca = PCA(n_components=min(self.x.shape[0], self.x.shape[1]))
        return pca, pca.fit_transform(self.x)

    """
    	SVD Dimensionality Reduction
    """

    def svd(self):
        svd = TruncatedSVD(n_components=self.k)
        return svd, svd.fit_transform(self.x)

    """
    	NMF Dimensionality Reduction
    """

    def nmf(self):
        nmf = NMF(n_components=self.k)
        return nmf, nmf.fit_transform(self.x)

    """
    	LDA Dimensionality Reduction
    """

    def lda(self):
        lda = LatentDirichletAllocation(n_components=self.k)
        return lda, lda.fit_transform(self.x)
