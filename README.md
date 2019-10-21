# HandImages

##Phase 1

This project is an implementation of two feature descriptors (Local Binary Patterns and SIFT) on an image dataset. The goal of the project is to identify similar images based on their respective feature descriptors. 

The project has three tasks as follows. 
1. Feature extraction of an image (LBP or SIFT) 
2. Storing feature descriptors of all the images in a folder/DB. (Feature descriptors are very large arrays. Storing them in a database made more sense than in flat files and makes task 3 computationally less intensive. 
3. Computing k most similar images based on feature descriptors

In order to execute these tasks, run `phase1/driver.py` and select any option out of the three. 

Requirements for the project:
1. `scikit-image`
2. `open-cv`
3. `pymongo`
4. `mongodb`

##Phase 2

This Phase is continuation on Phase 1. In this phase, we experimented with image features, vector models, and dimensionality reduction.

In total, there was 8 tasks.

1.	For all the images in provided Image Dataset, feature vector space was calculated from each of the four feature models namely (CM, LBP, HOG, SIFT) and is fitted into k latent space using all following dimensionality reduction techniques (PCA, SVD, NMF, LDA).

2.	For a given image, reported the most m similar images present in the Dataset along with their score using results of Task1.

3.	 Reported k latent semantics for all the images, this time extracted semantics corresponds to different labels provided in metadata for each image.

4.	For a given image, reported the most m similar images present in the Dataset along with their score using results of Task3.

5.	Classified an unlabeled image into pair of known labels known for Dataset such as left vs right, dorsal vs palmer.  

6.	For a given subject, visualize the most related three subjects using any of the feature model and reduction technique.

      Note: Subject here is not an image, but a person who can have multiple images present in the image Dataset. 
      
7.	Reported top k latent semantics for all the given subjects using dimensionality reduction technique NMF.
      Note: Finding k latent semantic here means finding k dominant subjects which are similar to a particular subject.

8.	Reported top k latent semantics in the image and metadata-space using NMF as dimensionality reduction technique.

    Note: Dimensionality Reduction was done on feature space which is matrix of images as row and their binary relation to metadata as       column.
    
9.	Visualized latent semantics in terms of Data and Feature.

In order to execute these tasks, run `phase2/driver.py` and select any option out of the 9.

Requirements for the Phase2:

1. ` scikit-image==0.15.0 `
2. ` scikit-learn==0.21.3 `
3. ` scipy==1.3.1 `
4. ` numpy==1.17.2 `.
5. ` opencv-contrib-python==3.4.2.16 `.
6. ` opencv-python==3.4.2.16 `.
7. ` pandas==0.24.2 `.
8. ` numpy==1.17.2 `.
9. ` matplotlib==3.0.3 `.
10.` pymongo==3.9.0 `.
11.` Pillow==6.1.0 `.
