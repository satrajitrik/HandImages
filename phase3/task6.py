import numpy
import functions, task5, visualizer
import svm
import visualizer
from kernel import Kernel
from database import Database
from ppr import PageRank


def get_feedback(similar_images):
    feedback = []
    if type(similar_images[0]) is list or type(similar_images[0]) is tuple:
        for id, _ in similar_images:
            response = input("Is {} relevant? y/n ".format(id))
            feedback.append((id, response))
    else:
        visualizer.visualize_feedback(similar_images,'SVM')
        feedback = visualizer.stored_values();
        # for id in similar_images:
        #     response = input("Is {} relevant? y/n ".format(id))
        #     feedback.append((id, response))

    return feedback


def new_relevantlist(relevant_list,irrelevant_list, test_images, label_test):
    a=0
    newimage_list=[]
    for items in relevant_list:
        newimage_list.append(items)

    for index, item in enumerate(label_test):
        if item == 1:
            newimage_list.append(test_images[index])
    newimage_list.extend(irrelevant_list)
    for index, item in enumerate(label_test):
        if item == -1:
            newimage_list.append(test_images[index])

    return newimage_list


def svm_feedback(image_id, all_images):
    test_images = []
    _list = []
    relevant_list, irrelevant_list = Database().retrieve_feedback(image_id)
    if type(all_images[0]) is list or type(all_images[0]) is tuple:
        for i, _ in all_images:
            if i not in relevant_list and i not in irrelevant_list:
                test_images.append(i)
    else:
        for i in all_images:
            if i not in relevant_list and i not in irrelevant_list:
                test_images.append(i)


    new_imagelist = relevant_list + irrelevant_list
    print("training images ", new_imagelist)
    print("Test images", test_images)
    feature_desc_train=Database().retrieve_many(image_ids=new_imagelist)

    training_data = numpy.array([item["vector"] for item in feature_desc_train])
    labels_training = [item["image_id"] for item in feature_desc_train]

    for i in range(0, len(labels_training)):
        if labels_training[i] in irrelevant_list:
            labels_training[i] = -1
        elif labels_training[i] in relevant_list:
            labels_training[i] = 1
    labels_training=numpy.array(labels_training)

    classifer = svm.binary_classification_smo(kernel=Kernel._polykernel(5))  # try 5 and 10 for dimensions in polykernel
    classifer.fit(training_data, labels_training)

    feature_desc_test = Database().retrieve_many(image_ids=test_images)

    test_data = numpy.array([item["vector"] for item in feature_desc_test])

    y = classifer.predict(test_data)
    print("Predicted classes by SVM Classifer: ", y)
    new_order = new_relevantlist(relevant_list, irrelevant_list, test_images, y)
    print('new Rank', new_order)
    return new_order


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


def ppr_based_feedback(page_rank, image_id):
    output_images = []
    relevant_images, irrelevant_images = Database().retrieve_feedback(image_id=image_id)
    page_rank.s = numpy.ones(page_rank.node_count)
    all_images = page_rank.all_image_names
    for image in relevant_images:
        page_rank.s[all_images.index(image)] = 2
    for image in irrelevant_images:
        page_rank.s[all_images.index(image)] = 0

    # Column normalize to one
    if sum(page_rank.s):
        page_rank.s = page_rank.s / sum(page_rank.s)
    queue = page_rank.perform_random_walk()
    while not queue.empty():
        probability, image = queue.get()
        probability = -probability
        if not probability and image not in irrelevant_images:
            output_images.append((image, probability))
    return output_images


def init_ppr(all_images):
    image_vectors = Database().retrieve_many([image[0] for image in set(all_images)])
    page_rank = PageRank(k=8, labelled_images=image_vectors)
    page_rank.generate_graph()
    return page_rank


def feedback_loop(image_id, similar_images, all_images, m, algorithm):
    response = input("Satisfied with search results? y/n ")
    similarimagesforsvm = []
    if response == "y":
        return

    if algorithm == 3:
        page_rank = init_ppr(all_images)

    if algorithm == 1:
        for id, _ in similar_images:
            similarimagesforsvm.append(id)
        feedback = get_feedback(similarimagesforsvm)
    else:
        feedback = get_feedback(similar_images)

    while 1:
        Database().store_feedback(image_id, feedback)
        if algorithm == 1:
            similar_images = svm_feedback(image_id, similar_images)
        elif algorithm == 2:
            pass
        elif algorithm == 3:
            similar_images = sorted(
            ppr_based_feedback(page_rank=page_rank, image_id=image_id),
            key=lambda x: x[1],
            reverse=True,
            )[:m]
        elif algorithm == 4:
            similar_images = sorted(
            probablistic_feedback(image_id, all_images),
            key=lambda x: x[1],
            reverse=True,
            )[:m]

        if algorithm ==1:
            visualizer.visualize_feedback(similar_images,'SVM')

        else:
            visualizer.visualize_lsh(image_id, similar_images)

        response = input("Satisfied with search results? y/n ")

        if response == "y":
            break
        if algorithm == 1:
            Database().delete_prev_feedback(image_id)

        feedback = get_feedback(similar_images)

    return similar_images


def starter(image_id, m, k, l, algorithm):
    similar_images, all_images = task5.starter(image_id, m, k, l)

    feedback_loop(image_id, similar_images, all_images, m, algorithm)