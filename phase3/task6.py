import functions, task5, visualizer

from database import Database


def get_feedback(similar_images):
    feedback = []
    for id, _ in similar_images:
        response = input("Is {} relevant? y/n ".format(id))
        feedback.append((id, response))

    return feedback


def probablistic_feedback(image_id, all_images):
    relevant_images = []
    relevant_list, irrelevant_list = Database().retrieve_feedback(image_id)

    for id, _ in all_images:
        """
        Comparing the priors based on Naive Bayes formula. P(B|A) = P(A|B).P(B) / P(A)
        We only need to calculate P(A|B) for comparison
        """
        relevance_prior = relevant_list.count(id) / len(relevant_list)
        irrelevance_prior = irrelevant_list.count(id) / len(irrelevant_list)

        if relevance_prior >= irrelevance_prior:
            relevant_images.append((id, relevance_prior))

    return relevant_images


def feedback_loop(image_id, similar_images, all_images, m, algorithm):
    response = input("Satisfied with search results? y/n ")

    if response == "y":
        return

    feedback = get_feedback(similar_images)

    while 1:
        Database().store_feedback(image_id, feedback)

        similar_images = sorted(
            probablistic_feedback(image_id, all_images),
            key=lambda x: x[1],
            reverse=True,
        )[:m]
        visualizer.visualize_lsh(image_id, similar_images)

        response = input("Satisfied with search results? y/n ")
        if response == "y":
            break

        feedback = get_feedback(similar_images)

    return


def starter(image_id, m, k, l, algorithm):
    similar_images, all_images = task5.starter(image_id, m, k, l)

    feedback_loop(image_id, similar_images, all_images, m, algorithm)
