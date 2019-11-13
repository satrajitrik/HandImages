from decisiontreeclassifier import DecisionTreeClassifier as dt
from database import Database
from config import Config
import visualizer
import pandas
import numpy as np


def starter():

    #retrive Training data from Database
    df = Database().retrieve_many(collection_type= "training")
    trainingdata = np.array(df["latent_symantics"].values.tolist())
    labels = df['label'].to_numpy()

    classifer = dt(3)
    classifer.fit(trainingdata, labels)

    df = Database().retrieve_many(collection_type= "testing")
    testingdata = np.array(df["latent_symantics"].values.tolist())

    y = classifer.predict(testingdata)
    print(y)


    m= len(y)
    images = df["image_id"].values.tolist()


    metadata = pandas.read_csv(Config().metadata_file())
    oy = []
    correct = 0
    for i in range(len(images)):
        imageName = images[i] + ".jpg"
        temp= metadata[metadata.imageName == imageName]['aspectOfHand'].values
        if(temp== 'dorsal right' or temp == "dorsal left"):
            oy.append(1)
        else:
            oy.append(0)
        if oy[i] == y[i]:
            correct+=1
    print(images)
    print(oy)
    print("correctly classified : ", correct)
    print("Incorrectly classified : ", (m - correct))
    print("accuracy : " , (correct * 1.0)/(m * 1.0))

    visualizer.task4visualizar(images,y)
