from scipy.spatial import distance


def calculate_similarity(source_vector, target_vector):
    return 1 / (1 + distance.euclidean(source_vector, target_vector) ** (0.25))


def find_similarity(source_vector, target_vectors, m):
    similarities = [
        (id, calculate_similarity(source_vector, vector))
        for id, vector in target_vectors
    ]

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:m]
