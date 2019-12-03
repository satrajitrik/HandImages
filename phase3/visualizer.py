import matplotlib.pyplot as plt
import tkinter as tk
import cv2
import os
import TkinterCheckbox as tc
from VerticalScrollableFrame import VSF
from PIL import Image
from PIL import ImageTk
from config import Config

img_dir = Config().read_testing_set1_path()
thumbnail_size = (160, 120)
symatics_width = 1600
data_symantics_height = 800


def var_status():

    lr = []
    lr1 = []
    for box in tc.CheckBox.boxes:
        if box.text == 'Relevant' and box.var.get():  # Checks if the button is ticked
            lr.append(box.var.get())
        if box.text == 'Irrelevant' and box.var.get():  # Checks if the button is ticked
            lr1.append(box.var.get())
    print(lr)
    print(lr1)
    records = {
        "Relevant": lr,
        "Irrelevant": lr1

    }


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


def create_thumbnail(img_id):
    # Load an image using OpenCV
    img_path = os.path.join(img_dir, img_id)
    # print('Loading image at path: %s' % img_path)
    cv_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    tn_img = cv2.resize(cv_img, thumbnail_size, interpolation=cv2.INTER_AREA)
    return tn_img


def visualize_svm_classifier(images_labeled, classifier_type):
    i = 0
    ldv1 = []
    for ls_list in images_labeled.values:
        for img, score in ls_list:
            ldv1.append(img)

    photos = []
    # Created a window
    window = tk.Tk()

    title_txt = (
            "Visualization of %s - Classifier"
            % classifier_type
    )
    window.title(title_txt)

    frame = VSF(window, symatics_width, data_symantics_height)

    v_row = 0
    img_col = 0
    lbl_col = 1
    ls_count = 1
    p_count = 0
    count=0
    for ls_list in images_labeled.values:
        v_row += 1
        if(count%12) == 0:
            v_row = 1
            img_col += 2
        for img, score in ls_list:
            row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            tn_img = create_thumbnail(img)
            height, width, no_channels = tn_img.shape
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(row, width=width, height=height)
            # Use PIL (Pillow) to convert the NumPy nd array to a PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
            photos.append(photo)
            canvas.create_image(0, 0, image=photos[p_count], anchor=tk.NW)

            match_label = tk.Label(frame.scrollable_frame, text=img)
            match_label.grid(row=v_row, column=img_col, columnspan=2)
            v_row += 1
            label = tk.Label(row, text=score)

            row.grid(row=v_row, column=img_col, columnspan=2)
            canvas.grid(row=v_row, column=img_col)
            label.grid(row=v_row, column=lbl_col)
            v_row += 1
            checkbox1 = tc.CheckBox(row, text='Relevant', fg='green',onvalue=ldv1[p_count], offvalue=None)
            checkbox2 = tc.CheckBox(row, text='Irrelevant',fg='red' ,onvalue=ldv1[p_count], offvalue=None)
            p_count += 1
            checkbox1.grid(row=v_row, column=lbl_col)
            checkbox2.grid(row=v_row, column=lbl_col+1)

            v_row += 1
            count += 3


        #v_row = 0
        # ls_count += 1
        # img_col += 2
        # lbl_col += 2

    quit_button = tk.Button(window, text="Quit", command=window.quit ,width='15', fg="red")
    quit_button.pack(side=tk.BOTTOM)
    save_button = tk.Button(window, text="Save", command=var_status, width='15', fg="blue")
    save_button.pack(side=tk.BOTTOM)
    frame.pack(side="left", fill="both")
    window.mainloop()

