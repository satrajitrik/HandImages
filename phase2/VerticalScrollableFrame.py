import tkinter as tk
from tkinter import ttk


class VSF(ttk.Frame):
    def __init__(self, container, width, height):
        super().__init__(container)
        canvas = tk.Canvas(self, width=width, height=height)
        xscrollbar = ttk.Scrollbar(self, orient="horizontal", command=canvas.xview)
        yscrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)

        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(xscrollcommand=xscrollbar.set)
        canvas.configure(xscrollcommand=yscrollbar.set)

        yscrollbar.pack(side="right", fill="y")
        xscrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="top", fill="both", expand=True)


