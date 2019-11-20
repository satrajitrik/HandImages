from queue import PriorityQueue

import numpy

from database import Database
import functions


class PageRank(object):
    def __init__(self, image_vectors, k):
        self.k = k
        self.image_vectors = image_vectors
        self.node_count = len(image_vectors)
        print(self.node_count)
        self.graph = numpy.zeros((self.node_count, self.node_count))   # Transpose of Adjacency matrix
        self.similarity_matrix = numpy.zeros((self.node_count, self.node_count))
        self.s = None   # Teleport vector
        self.c = 0.15   # (1 - alpha)
        self.matrix_inverse = None
        self.steady_state_prob_vector = None    # PI_steady_state
        self.image_names = []
    
    def generate_graph(self):
        self.node_count = len(self.image_vectors)
        # Generate similarity graph
        self.image_names = []
        for i in range(self.node_count):
            self.image_names.append(self.image_vectors[i]['image_id'])
        for i in range(self.node_count):
            for j in range(i+1, self.node_count):
                # Compute Image similarity
                similarity = functions.calculate_similarity(self.image_vectors[i]['vector'], self.image_vectors[j]['vector'])
                self.similarity_matrix[i][j] = similarity
                self.similarity_matrix[j][i] = similarity
        
        for i in range(self.node_count):
            self.image_vectors.append(self.image_vectors[i]['image_id'])
            
        #   Set top k edges from the image
        self.graph = numpy.zeros((self.node_count, self.node_count))
        for j in range(self.node_count):
            p = PriorityQueue()
            for i in range(self.node_count):
                p.put((-self.similarity_matrix[i][j], i))
            # self.graph[j] = numpy.zeros(self.node_count)  # Reset the row to 0
            for k in range(self.k):     # Get top k elements from the row
                element = p.get()
                self.graph[element[1]][j] = -element[0]
        
        #   Normalize the columns
        for i in range(self.node_count):
            column_sum = numpy.sum(self.graph[:, i])
            self.graph[:, i] = self.graph[:, i] / float(column_sum)
        Database().insert_binary(key=str(self.node_count), value=self.graph, collection_type="page_rank")
        return self.graph
    
    def recompute_graph(self):
        pass
    
    def get_graph(self):
        self.graph = Database().retrieve_binary(key=str(self.node_count), collection_type="page_rank")
        return self.graph
    
    def compute_matrix_inverse(self, recompute=False):
        if self.matrix_inverse is None or recompute:
            self.matrix_inverse = numpy.linalg.inv(numpy.subtract(numpy.identity(self.node_count),
                                                                  (1-self.c) * self.graph))
        return self.matrix_inverse

    def perform_random_walk(self, query_images):
        """
        :param query_image_positions: Position of the query image in the graph (Restart position)
        :return:
        """
        try:
            query_image_positions = [self.image_names.index(x) for x in query_images]
        except ValueError:
            print("Query images: ", query_images)
            print("Images available: ", self.image_names)
            raise ValueError("Query image not available in the list")
        self.s = numpy.zeros(self.node_count)
        for position in query_image_positions:
            self.s[position] = 1.0 / len(query_image_positions)
        self.compute_matrix_inverse()
        # print(self.s)
        self.steady_state_prob_vector = numpy.matmul(self.matrix_inverse, self.c * self.s)
        p = PriorityQueue()
        for i in range(self.node_count):
            p.put((-self.steady_state_prob_vector[i], self.image_names[i]))
        return p
