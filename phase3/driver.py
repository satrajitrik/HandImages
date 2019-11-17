from phase3 import task3, task4, task5


def main():
    task = int(input("Input Task number:"))
    if task == 3:
        task3.starter()
    elif task == 4:
        task4.starter()
    elif task == 5:
        image_id = input("Enter the image ID: ")
        m = int(input("Enter m: "))
        k = int(input("Enter k: "))
        l = int(input("Enter l: "))
        task5.starter(image_id, m, k, l)


if __name__ == "__main__":
    main()
