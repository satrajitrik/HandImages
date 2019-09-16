# HandImages
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

To be continued .......
