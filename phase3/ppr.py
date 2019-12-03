from queue import PriorityQueue

import numpy

from database import Database
import functions


class PageRank(object):
    def __init__(self, k, labelled_images, unlabelled_images=None):
        self.k = k
        self.s = None  # Teleport vector
        self.c = 0.15  # (1 - alpha)
        self.matrix_inverse = None
        self.steady_state_prob_vector = None  # PI_steady_state
        self.image_names = []  # List of image names (Labelled folder)
        self.image_vectors = labelled_images
        self.unlabelled_image_vectors = unlabelled_images
        self.all_image_vectors = []
        self.node_count = len(self.image_vectors)
        self.all_image_vectors.extend(self.image_vectors)
        if unlabelled_images:
            self.all_image_vectors.extend(self.unlabelled_image_vectors)
            self.node_count += len(self.unlabelled_image_vectors)
        if self.k > self.node_count:
            raise ValueError("k is greater than number of nodes")
        print(self.node_count)

        self.graph = numpy.zeros(
            (self.node_count, self.node_count)
        )  # Transpose of Adjacency matrix
        self.similarity_matrix = numpy.zeros((self.node_count, self.node_count))

    def generate_graph(self):
        # Task 3
        # Generate similarity graph
        self.image_names = []
        self.all_image_names = []
        for i in range(self.node_count):
            self.image_names.append(self.image_vectors[i]["image_id"])
        self.all_image_names.extend(self.image_names)
        for i in range(self.node_count):
            for j in range(i + 1, self.node_count):
                # Compute Image similarity
                similarity = functions.calculate_similarity(
                    self.image_vectors[i]["vector"], self.image_vectors[j]["vector"]
                )
                self.similarity_matrix[i][j] = similarity
                self.similarity_matrix[j][i] = similarity

        #   Set top k edges from the image
        self.graph = numpy.zeros((self.node_count, self.node_count))
        for j in range(self.node_count):
            p = PriorityQueue()
            for i in range(self.node_count):
                p.put((-self.similarity_matrix[i][j], i))
            # self.graph[j] = numpy.zeros(self.node_count)  # Reset the row to 0
            for k in range(self.k):  # Get top k elements from the row
                element = p.get()
                self.graph[element[1]][j] = -element[0]

        #   Normalize the columns
        for i in range(self.node_count):
            column_sum = numpy.sum(self.graph[:, i])
            self.graph[:, i] = self.graph[:, i] / float(column_sum)
        Database().insert_binary(
            key=str(self.node_count), value=self.graph, collection_type="page_rank"
        )
        return self.graph

    def label_images(self):
        self.image_names = []  # List of labelled folder image names
        self.unlabelled_image_names = []  # List of unlabelled folder image names
        self.all_image_names = []  # List of labelled + unlabelled image names
        self.is_labelled = {}  # ImageID -> True/False
        self.new_labels = {}  # ImageID -> Label - "dorsal"/"palmar" (Unlabelled Image)
        self.labels = {}  # ImageID -> Label 1 (dorsal)/ 0 (palmar) - (Labelled Image)

        for i in range(len(self.image_vectors)):
            self.image_names.append(self.image_vectors[i]["image_id"])
            self.all_image_names.append(self.image_vectors[i]["image_id"])
            self.is_labelled[self.image_vectors[i]["image_id"]] = True
            self.labels[self.image_vectors[i]["image_id"]] = self.image_vectors[i][
                "label"
            ]

        for i in range(len(self.unlabelled_image_vectors)):
            self.unlabelled_image_names.append(
                self.unlabelled_image_vectors[i]["image_id"]
            )
            self.is_labelled[self.unlabelled_image_vectors[i]["image_id"]] = False
            self.all_image_names.append(self.unlabelled_image_vectors[i]["image_id"])

        for i in range(self.node_count):
            for j in range(i + 1, self.node_count):
                # Compute Image similarity
                similarity = functions.calculate_similarity(
                    self.all_image_vectors[i]["vector"],
                    self.all_image_vectors[j]["vector"],
                )
                self.similarity_matrix[i][j] = similarity
                self.similarity_matrix[j][i] = similarity

        # Set top k edges from the image
        self.graph = numpy.zeros((self.node_count, self.node_count))
        for j in range(self.node_count):
            p = PriorityQueue()
            for i in range(self.node_count):
                p.put((-self.similarity_matrix[i][j], i))
            # self.graph[j] = numpy.zeros(self.node_count)  # Reset the row to 0
            for k in range(self.k):  # Get top k elements from the row
                element = p.get()
                self.graph[element[1]][j] = -element[0]

        # Normalize the columns
        for i in range(self.node_count):
            column_sum = numpy.sum(self.graph[:, i])
            self.graph[:, i] = self.graph[:, i] / float(column_sum)

        # print(self.compute_matrix_inverse())
        for image_name in self.unlabelled_image_names:
            probability_vector = self.perform_random_walk([image_name])
            palmar_probability, dorsal_probability = 0.0, 0.0
            # print(probability_vector)
            while not probability_vector.empty():
                probability, image = probability_vector.get()
                probability = -probability
                if not probability:
                    break
                # print(probability, image, self.is_labelled[image])
                if self.is_labelled[image]:
                    if self.labels[image] == 1:  # Dorsal
                        dorsal_probability += probability
                    else:
                        palmar_probability += probability
            # print(self.labels.values())
            mean_dorsal_probability = dorsal_probability / list(
                self.labels.values()
            ).count(1)
            mean_palmar_probability = palmar_probability / list(
                self.labels.values()
            ).count(0)
            # print(palmar_probability, dorsal_probability,)
            self.new_labels[image_name] = (
                "dorsal"
                if mean_dorsal_probability > mean_palmar_probability
                else "palmar"
            )
            # print()
            # print(image_name, self.new_labels[image_name])
        return self.new_labels

    def get_graph(self):
        self.graph = Database().retrieve_binary(
            key=str(self.node_count), collection_type="page_rank"
        )
        return self.graph

    def compute_matrix_inverse(self, recompute=False):
        if self.matrix_inverse is None or recompute:
            self.matrix_inverse = numpy.linalg.inv(
                numpy.subtract(
                    numpy.identity(self.node_count), (1 - self.c) * self.graph
                )
            )
        return self.matrix_inverse

    def perform_random_walk(self, query_images=None):
        """
        :param query_images: Position of the query image in the graph (Restart position)
        :return:
        """
        if query_images:
            # Set teleport matrix using query image
            try:
                query_image_positions = [
                    self.all_image_names.index(x) for x in query_images
                ]
            except ValueError:
                print("Query images: ", query_images)
                print("Images available: ", self.all_image_names)
                raise ValueError("Query image not available in the list")
            self.s = numpy.zeros(self.node_count)
            for position in query_image_positions:
                self.s[position] = 1.0 / len(query_image_positions)
        else:
            # Set teleport matrix from outside
            pass
        self.compute_matrix_inverse()
        # print(self.s)
        self.steady_state_prob_vector = numpy.matmul(
            self.matrix_inverse, self.c * self.s
        )
        # print(self.steady_state_prob_vector)
        p = PriorityQueue()
        for i in range(self.node_count):
            p.put((-self.steady_state_prob_vector[i], self.all_image_names[i]))
        return p
