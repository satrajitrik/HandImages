from queue import PriorityQueue

import numpy


class PageRank(object):
    def __init__(self, image_vectors, k):
        self.k = k
        self.image_vectors = image_vectors
        self.node_count = len(image_vectors)
        self.graph = numpy.zeros(self.node_count)   # Transpose of Adjacency matrix
        self.s = None   # Teleport vector
        self.c = 0.15   # (1 - alpha)
        self.matrix_inverse = None
        self.steady_state_prob_vector = None    # PI
    
    def generate_graph(self):
        self.node_count = len(self.image_vectors)
        # Generate similarity graph
        for i in range(self.node_count):
            for j in range(i+1, self.node_count):
                # Compute Image similarity
                similarity = 0
                self.graph[i][j] = similarity
                self.graph[j][j] = similarity
        
        #   Set top k edges from the image
        for i in range(self.node_count):
            p = PriorityQueue()
            for j in range(self.node_count):
                p.put((-self.graph[i][j], j))
            self.graph[i] = numpy.zeros(self.node_count)  # Reset the row to 0
            for k in range(self.k):     # Get top k elements from the row
                element = p.get()
                self.graph[i][element[1]] = -element[0]
        
        #   Normalize the columns
        for i in range(self.node_count):
            column_sum = numpy.sum(self.graph[:, i])
            self.graph[:, i] = self.graph[:, i] / float(column_sum)
        
        return self.graph
    
    def recompute_graph(self):
        pass
    
    def get_graph(self):
        return self.graph
    
    def compute_matrix_inverse(self, recompute=False):
        if self.matrix_inverse is None or recompute:
            self.matrix_inverse = numpy.linalg.inv(numpy.subtract(numpy.identity(self.node_count),
                                                                  (1-self.c) * self.graph))
        return self.matrix_inverse

    def perform_random_walk(self, query_image_position):
        """
        :param query_image_position: Position of the query image in the graph (Restart position)
        :return:
        """
        self.s = numpy.zeros((self.node_count, 1))
        self.s[query_image_position] = 1
        self.compute_matrix_inverse()
        self.steady_state_prob_vector = numpy.multiply(self.matrix_inverse, self.c * self.s)
