class Labels(object):
    def __init__(self, label_choice):
        self.label = self._get_label(label_choice)

    def _get_label(self, label_choice):
        if label_choice == 1:
            return ("left_hand", 1)
        elif label_choice == 2:
            return ("left_hand", 0)
        elif label_choice == 3:
            return ("dorsal", 1)
        elif label_choice == 4:
            return ("dorsal", 0)
        elif label_choice == 5:
            return ("accessories", 1)
        elif label_choice == 6:
            return ("accessories", 0)
        elif label_choice == 7:
            return ("male", 1)
        else:
            return ("male", 0)
