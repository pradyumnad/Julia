# Loading Data

import numpy as np

import pandas as pd
from skimage.exposure import exposure
from skimage.filters import threshold_otsu
from skimage.io import imread, imshow
from sklearn.cross_validation import cross_val_score as k_fold_CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
import time


def read_data(typeData, labelsInfo, imageSize, path):
    print(path, imageSize)
    # Intialize x  matrix
    x = np.zeros((labelsInfo.shape[0], imageSize*2))

    for (index, idImage) in enumerate(labelsInfo["ID"]):
        # Read image file
        nameFile = "{0}/{1}Resized20/{2}.Bmp".format(path, typeData, idImage)
        img = imread(nameFile, as_grey=True)

        thresh = threshold_otsu(img)
        binary = img > thresh
        # imshow(binary)

        # edges2 = feature.canny(img)
        img_eq = exposure.equalize_hist(img, nbins=64)
        pixels = np.reshape(binary, (1, imageSize))
        pixels2 = np.reshape(img_eq, (1, imageSize))

        # print(img_eq)
        # imshow(img_eq)

        x[index, :] = np.append(pixels, pixels2)
    return x


imageSize = 400  # 100 x 100 pixels

# Set location of data files , folders
path = "/Users/pradyumnad/GitHub/Julia"

labelsInfoTrain = pd.read_csv("{0}/trainLabels.csv".format(path))

# Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

yTrain = map(ord, labelsInfoTrain["Class"])
yTrain = np.array(yTrain)

# Importing main functions

from sklearn.neighbors import KNeighborsClassifier as KNN


# Running LOOF-CV with 3NN sequentially

# start = time.time()
# model = KNN(n_neighbors=3, weights='distance')
# cvAccuracy = np.mean(k_fold_CV(model, xTrain, yTrain, cv=5, scoring="accuracy"))
# print("The 2-CV accuracy of 3NN", cvAccuracy)
# print(time.time() - start, "seconds elapsed")

# Tuning the value for k

start = time.time()
model = KNN(n_neighbors=3, weights='distance')
tuned_parameters = [{"n_neighbors": list([1, 5, 10])}]
clf = GridSearchCV(model, tuned_parameters, cv=5, scoring="accuracy", verbose=2)
clf.fit(xTrain, yTrain)
print(clf.grid_scores_)
print(time.time() - start, "seconds elapsed")
#
model = RandomForestClassifier(max_features=29, n_estimators=250, criterion="entropy", n_jobs=-1)
# cvAccuracy = np.mean(k_fold_CV(model, xTrain, yTrain, cv=2, scoring="accuracy", verbose=2))
# print("Acc: ", cvAccuracy)

tuned_param = [{"n_estimators": list([50, 250])}]
clf = GridSearchCV(model, tuned_param, cv=5, scoring="accuracy", verbose=2)
clf.fit(xTrain, yTrain)
print(clf.grid_scores_)


def test():
    # Read information about test data ( IDs ).
    labelsInfoTest = pd.read_csv("{0}/sampleSubmission.csv".format(path))

    # Read test matrix
    xTest = read_data("test", labelsInfoTest, imageSize, path)

    # model = KNN(n_neighbors=5)
    model = RandomForestClassifier(max_features=29, n_estimators=350, criterion="entropy", n_jobs=-1)
    model.fit(xTrain, yTrain)
    print("Predicting..")
    yTest = model.predict(xTest)
    yTest = map(chr, yTest)
    labelsInfoTest["Class"] = yTest

    labelsInfoTest.to_csv("juliaRFSubmission.csv", index=False)
    print("Wrote")
    # write it
    # writer = open('juliaKNNSubmission.csv', 'w')
    # writer.write("ID,Class")
    # for i in range(0, len(yTest)):
    #     print(labelsInfoTest["ID"][i], yTest[i])
    #     writer.write(labelsInfoTest["ID"][i] + "," + labelsInfoTest["Class"][i])

# test()
