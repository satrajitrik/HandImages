import matplotlib.pyplot as plt

from config import Config


def visualize_lsh(source_image_id, similar_images):
    source_axis = plt.subplot2grid(
        (2, len(similar_images)), (0, 0), colspan=len(similar_images)
    )

    read_all_path = Config().read_all_path()
    source_image = plt.imread("{}{}.jpg".format(read_all_path, source_image_id))
    source_axis.imshow(source_image)
    source_axis.set_xlabel("Source Image: {}".format(source_image_id))

    for i, id_value_pair in enumerate(similar_images):
        image = plt.imread("{}{}.jpg".format(read_all_path, id_value_pair[0]))
        similar_axis = plt.subplot2grid((2, len(similar_images)), (1, i), colspan=1)
        similar_axis.imshow(image)
        similar_axis.set_xlabel(
            "{}, {}".format(id_value_pair[0], round(id_value_pair[1], 3))
        )

    plt.show()


def visualize_task4(images, y, classifier):
    m = len(y)
    col = 4
    row = int(m / col) if m % col == 0 else int(m / col) + 1
    fig, axes = plt.subplots(row, col)
    ax = axes.ravel()
    for i in range(m):
        ax[i].imshow(
            plt.imread(Config().read_all_path() + images[i] + ".jpg"),
            interpolation="none",
        )
        # print(y[i],)
        l = "Dorsal" if (y[i] == 1 or y[i] == "dorsal") else "Palmar"
        ax[i].set_title(l, fontsize=8)
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)

    for i in range(m, row * col):
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)

    fig.suptitle("Image Classification using " + classifier)
    plt.show()


def visualize_task3(images_probability_pair):
    m = len(images_probability_pair)
    col = 3
    row = int(m / col) if m % col == 0 else int(m / col) + 1
    fig, axes = plt.subplots(row, col)
    ax = axes.ravel()
    for i, (image, probability) in enumerate(images_probability_pair):
        ax[i].imshow(
            plt.imread(Config().read_all_path() + image + ".jpg"), interpolation="none"
        )
        ax[i].set_title(probability, fontsize=8)
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)

    # for i in range(m, row * col):
    #     ax[i].xaxis.set_visible(False)
    #     ax[i].yaxis.set_visible(False)

    fig.suptitle("Ranked Images")
    plt.subplots_adjust(hspace=0.5)
    plt.show()
