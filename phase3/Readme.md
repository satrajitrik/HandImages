In this phase, we have 6 tasks.

Task1: Given a folder with unlabeled images, the system labels them as dorsal-hand vs palmar-hand
To run this task, just run ```driver.py``` and select ```task``` as 1.

Task2: Given a folder with unlabeled images, the system labels them as dorsal-hand vs palmar-hand
To run this task, just run ```driver.py``` and select ```task``` as 2.

Task3: Implement a program which, given a value k, creates an image-image similarity graph, such that from each
image, there are k outgoing edges to k most similar/related images to it. Given 3 user specified imageids on the graph,
the program identifies and visualizes K most dominant images using Personalized Page Rank (PPR) for a user supplied
K.
To run this task, just run ```driver.py``` and select ```task``` as 3 and enter corresponding inputs.

Task4: Implement a program which, given a folder with dorsal/palmar labeled images,
– creates an SVM classifer,
– creates a decision-tree classifier,
– creates a PPR based clasifier,
and, given a folder with unlabeled images, the system labels them as dorsal-hand vs palmar-hand
To run this task, just run ```driver.py``` and select ```task``` as 4 and enter corresponding inputs.

Task5: Implement a Locality Sensitive Hashing (LSH) tool (for Euclidean distance) which takes as input (a) the number
of layers, L, (b) the number of hashes per layer, k, and (c) a set of vectors as input and creates an in-memory index
structure containing the given set of vectors and given a query image and integer t, visualizes the t most similar images (also outputs the numbers of unique and overall number of images considered).
To run this task, just run ```driver.py``` and select ```task``` as 5 and enter corresponding inputs.

Task6: Implement a
- SVM based relevance feedback system,
– a decision-tree based relevance feedback system,
– a PPR-based relevance feedback system,
– a probabilistic relevance feedback systemwhich enable the user to label some of the results returned by 5b as relevant or irrelevant and then return a new set of ranked results, relying on the feedback system selected by the user, either by revising the query or by re-ordering the existing results.
To run this task, just run ```driver.py``` and select ```task``` as 6 and enter corresponding inputs.
