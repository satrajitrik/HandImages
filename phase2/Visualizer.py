import tkinter as tk
import cv2
import os
from VerticalScrollableFrame import VSF
from HorizontalScrollableFrame import HSF
from PIL import Image
from PIL import ImageTk
from config import Config

img_dir = Config().read_path()
thumbnail_size = (160, 120)
symatics_width = 1200
data_symantics_height = 600
ftr_symatics_height = 200


def create_thumbnail(img_id):
    # Load an image using OpenCV
    img_path = os.path.join(img_dir, img_id)
    # print('Loading image at path: %s' % img_path)
    cv_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    tn_img = cv2.resize(cv_img, thumbnail_size, interpolation=cv2.INTER_AREA)
    return tn_img


def visualize_data_symantics(data_symnatics, symantics_type, descriptor_type):
    """ Function to visualize the Data to Latent Semantics Matrix """
    photos = []
    # Created a window
    window = tk.Tk()
    title_txt = (
        "Visualization of Data-Latent Semantics for %s with %s Feature Descriptors"
        % (symantics_type, descriptor_type)
    )
    window.title(title_txt)

    frame = VSF(window, symatics_width, data_symantics_height)

    v_row = 0
    img_col = 0
    lbl_col = 1
    ls_count = 1
    p_count = 0
    for ls_list in data_symnatics.values:
        ls_label = tk.Label(
            frame.scrollable_frame, text="Latent Semantic %s" % ls_count
        )
        ls_label.grid(row=v_row, column=img_col, columnspan=2)
        v_row += 1
        for img, score in ls_list:
            row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            tn_img = create_thumbnail(img)
            height, width, no_channels = tn_img.shape
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(row, width=width, height=height)
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
            photos.append(photo)
            canvas.create_image(0, 0, image=photos[p_count], anchor=tk.NW)
            p_count += 1

            match_label = tk.Label(frame.scrollable_frame, text=img)
            match_label.grid(row=v_row, column=img_col, columnspan=2)
            v_row += 1
            label = tk.Label(row, text=str(round(score, 8)))
            row.grid(row=v_row, column=img_col, columnspan=2)
            canvas.grid(row=v_row, column=img_col)
            label.grid(row=v_row, column=lbl_col)
            v_row += 1

        v_row = 0
        ls_count += 1
        img_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill="both")
    window.mainloop()


def visualize_feature_symantics(feature_symnatics, symantics_type, descriptor_type):
    """ Function to visualize the Feature to Latent Semantics Matrix."""
    # Create a window
    window = tk.Tk()
    title_txt = (
        "Visualization of Feature-Latent Semantics for %s with %s Feature Descriptors"
        % (symantics_type, descriptor_type)
    )
    window.title(title_txt)

    frame = VSF(window, symatics_width, ftr_symatics_height)

    v_row = 0
    ftr_col = 0
    lbl_col = 1
    ls_count = 1
    for ls_list in feature_symnatics.values:
        ls_label = tk.Label(
            frame.scrollable_frame, text="Latent Semantic %s" % ls_count
        )
        ls_label.grid(row=v_row, column=ftr_col, columnspan=2)
        v_row += 1
        row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        feature_id = tk.Label(row, text="Image Identifier", width=15)
        score_id = tk.Label(row, text="Feature Score", width=15)
        row.grid(row=v_row, column=ftr_col, columnspan=2)
        feature_id.grid(row=v_row, column=ftr_col)
        score_id.grid(row=v_row, column=lbl_col)
        v_row += 1
        for feature, score in ls_list:
            data_row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            feature_label = tk.Label(data_row, text=str(feature), width=14)
            score_label = tk.Label(data_row, text=str(round(score, 8)), width=16)
            data_row.grid(row=v_row, column=ftr_col, columnspan=2)
            feature_label.grid(row=v_row, column=ftr_col)
            score_label.grid(row=v_row, column=lbl_col)
            v_row += 1

        v_row = 0
        ls_count += 1
        ftr_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill="both")
    window.mainloop()
