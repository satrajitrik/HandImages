import task1, task2, task3


def main():
	inp = int(input("\
		What do you want to do? \n\
			1. Feature extraction of an image (LBP or SIFT) \n\
			2. Store feature descriptors of all the images in a folder \n\
			3. Visualize k most similar images based on feature descriptors.......... \n\
		Select one from above: ")
	)

	if inp == 1:
		task1.starter()
	elif inp == 2:
		task2.starter()
	else:
		task3.starter()

if __name__ == "__main__":
	main()