class Labels(object):
    def __init__(self, label_choice):
        self.label = self._tupleize_label(label_choice)

    """
        Returns a tuple: tu
        tu[0] = label
        tu[1] = indicator if label is set
        tu[2] = indicator if complementary label is set
    """

    def _tupleize_label(self, label_choice):
        if label_choice == 1:
            return ("left_hand", 1, 0)
        elif label_choice == 2:
            return ("left_hand", 0, 1)
        elif label_choice == 3:
            return ("dorsal", 1, 0)
        elif label_choice == 4:
            return ("dorsal", 0, 1)
        elif label_choice == 5:
            return ("accessories", 1, 0)
        elif label_choice == 6:
            return ("accessories", 0, 1)
        elif label_choice == 7:
            return ("male", 1, 0)
        elif label_choice == 8:
            return ("male", 0, 1)
        else:
            return (None, None, None)

    def _detupleize_label(self, tupled_label):
        if tupled_label == ("left_hand", 1):
            return "left hand"
        elif tupled_label == ("left_hand", 0):
            return "right hand"
        elif tupled_label == ("dorsal", 1):
            return "dorsal"
        elif tupled_label == ("dorsal", 0):
            return "palmar"
        elif tupled_label == ("accessories", 1):
            return "with accessories"
        elif tupled_label == ("accessories", 0):
            return "without accessories"
        elif tupled_label == ("male", 1):
            return "male"
        elif tupled_label == ("male", 0):
            return "female"
        else:
            return "Wrong tuple format.... "
