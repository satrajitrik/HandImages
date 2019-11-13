import cv2
import matplotlib.pyplot as plt

from config import Config


def visualize_lsh(source_image_id, similar_images):
    source_axis = plt.subplot2grid(
        (2, len(similar_images)), (0, 0), colspan=len(similar_images)
    )

    read_path = Config().read_path()
    source_image = cv2.imread("{}{}.jpg".format(read_path, source_image_id))
    source_rgb_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    source_axis.imshow(source_rgb_image)
    source_axis.set_xlabel("Source Image: {}".format(source_image_id))

    for i, id_value_pair in enumerate(similar_images):
        image = cv2.imread("{}{}.jpg".format(read_path, id_value_pair[0]))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        similar_axis = plt.subplot2grid((2, len(similar_images)), (1, i), colspan=1)
        similar_axis.imshow(rgb_image)
        similar_axis.set_xlabel(
            "{}, {}".format(id_value_pair[0], round(id_value_pair[1], 3))
        )

    plt.show()

def task4visualizar(images,y):
    m = len(y)
    col = 6
    if(m%col == 0):
        row = int(m/col)
    else:
        row = int(m/col) + 1
    fig, axes = plt.subplots(row, col)
    ax = axes.ravel()
    for i in range(m):
        ax[i].imshow(plt.imread(Config().read_testing_data_path() + images[i] + ".jpg"),  interpolation='none')
        if(y[i] == 1): l = "Dorsal"
        else: l = "Palmar"
        ax[i].set_title(l ,fontsize = 10)
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)

    for i in range(m,row*col):
        ax[i].xaxis.set_visible(False)
        ax[i].yaxis.set_visible(False)

    fig.suptitle('Image Classification using Decision Tree')
    plt.show()
