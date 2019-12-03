import tkinter


class CheckBox(tkinter.Checkbutton):
    boxes = []  # Storage for all buttons

    def __init__(self, master=None, **options):
        tkinter.Checkbutton.__init__(self, master, options)
        self.boxes.append(self)
        self.var = tkinter.StringVar()  # var used to store checkbox state (on/off)
        self.text = self.cget('text')  # store the text for later
        self.onvalue = self.cget('onvalue')
        self.offvalue = self.cget('offvalue')
        self.configure(variable=self.var)