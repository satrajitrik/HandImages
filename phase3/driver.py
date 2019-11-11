import task5
import task4

def main():
    task = int(input("Input Task number:"))
    if task == 4:
       task4.starter()
    if task == 5:
        image_id = input("Enter the image ID: ")
        m = int(input("Enter m: "))
        task5.starter(image_id, m)


if __name__ == "__main__":
    main()
