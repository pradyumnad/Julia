# Loading Data

import numpy as np

import pandas as pd
from skimage.exposure import exposure
from skimage.io import imread


def read_data(typeData, labelsInfo, imageSize, path):
    print(path, imageSize)
    # Intialize x  matrix
    x = np.zeros((labelsInfo.shape[0], imageSize))

    for (index, idImage) in enumerate(labelsInfo["ID"]):
        # Read image file
        nameFile = "{0}/{1}Resized20/{2}.Bmp".format(path, typeData, idImage)
        img = imread(nameFile, as_grey=True)
        # edges2 = feature.canny(img)
        img_eq = exposure.equalize_hist(img, nbins=64)
        # print(img_eq)
        # imshow(edges2)

        x[index, :] = np.reshape(img_eq, (1, imageSize))
    return x


imageSize = 400  # 100 x 100 pixels

# Set location of data files , folders
path = ".."

labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

# Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

yTrain = map(ord, labelsInfoTrain["Class"])
yTrain = np.array(yTrain)

# Importing main functions

from sklearn.neighbors import KNeighborsClassifier as KNN


# Running LOOF-CV with 3NN sequentially

# start = time.time()
# model = KNN(n_neighbors=3)
# cvAccuracy = np.mean(k_fold_CV(model, xTrain, yTrain, cv=5, scoring="accuracy"))
# print("The 2-CV accuracy of 3NN", cvAccuracy)
# print(time.time() - start, "seconds elapsed")

# start = time.time()
# clf = RandomForestClassifier(n_estimators=30)
# # clf = clf.fit(xTrain, yTrain)
#
# cvAccuracy = np.mean(k_fold_CV(clf, xTrain, yTrain, cv=2, scoring="accuracy"))
# print("The RF accuracy", cvAccuracy)
# print(time.time() - start, "seconds elapsed")
#
# start = time.time()
# clf = ExtraTreeClassifier()
# # clf = clf.fit(xTrain, yTrain)
#
# cvAccuracy = np.mean(k_fold_CV(clf, xTrain, yTrain, cv=2, scoring="accuracy"))
# print("The ET accuracy", cvAccuracy)
# print(time.time() - start, "seconds elapsed")

# Tuning the value for k

# start = time.time()
# tuned_parameters = [{"n_neighbors": list([5, 10])}]
# clf = GridSearchCV(model, tuned_parameters, cv=5, scoring="accuracy")
# clf.fit(xTrain, yTrain)
# print(clf.grid_scores_)
# print(time.time() - start, "seconds elapsed")


def test():
    # Read information about test data ( IDs ).
    labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(path))

    # Read test matrix
    xTest = read_data("test", labelsInfoTest, imageSize, path)

    model = KNN(n_neighbors=5)
    model.fit(xTrain, yTrain)
    print("Predicting..")
    yTest = model.predict(xTest)
    yTest = map(chr, yTest)
    labelsInfoTest["Class"] = yTest

    labelsInfoTest.to_csv("juliaKNNSubmission.csv", index=False)
    print("Wrote")
    # write it
    # writer = open('juliaKNNSubmission.csv', 'w')
    # writer.write("ID,Class")
    # for i in range(0, len(yTest)):
    #     print(labelsInfoTest["ID"][i], yTest[i])
    #     writer.write(labelsInfoTest["ID"][i] + "," + labelsInfoTest["Class"][i])


test()
