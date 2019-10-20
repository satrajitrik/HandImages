from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD


class FLS(object):
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
            pca= PCA(n_components=self.k)
            transforemd=pca.fit_transform(self.x)
            return pca.components_
        pca= PCA(n_components=min(self.x.shape[0], self.x.shape[1]))
        transforemd=pca.fit_transform(self.x);
        return pca.components_

    """
    	SVD Dimensionality Reduction
    """

    def svd(self):
        svd = TruncatedSVD(n_components=self.k)
        transformed= svd.fit_transform(self.x)
        return svd.components_

    """
    	NMF Dimensionality Reduction
    """

    def nmf(self):
        nmf = NMF(n_components=self.k)
        transformed=nmf.fit_transform(self.x)
        return nmf.components_

    """
    	LDA Dimensionality Reduction
    """

    def lda(self):
        lda = LatentDirichletAllocation(n_components=self.k)
        transformed= lda.fit_transform(self.x)
        return lda.components_
