import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import cv2


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()


def main():
    x = []
    y = []
    imagePaths = sorted(list(paths.list_images('train')))
    counter = 500
    for image in imagePaths:
        x.append(extract_histogram(cv2.imread(image)))
        if counter > 0:
            y.append(0)
        else:
            y.append(1)
        counter -= 1
    #TODO ИЗМЕНИТЬ ЗНАЧЕНИЯ НИЖЕ
    clf = LinearSVC(C=0.94, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=2)

    clf.fit(X_train, y_train)
    #TODO ТУТ ТОЖЕ ИЗМЕНИТЬ
    print("Theta(281): ", clf.coef_[0][280]) # T(280)
    print("Theta(131): ", clf.coef_[0][130]) # T(130)
    print("Theta(442): ", clf.coef_[0][441]) # T(441)
    y_predicted = clf.predict(X_test)

    f1 = f1_score(y_test, y_predicted, average=None)
    print("Average Macro-F1 score: ", ((f1[0] + f1[1])/2))

    x_new = []
    imagePaths2 = sorted(list(paths.list_images('test')))
    for image in imagePaths2:
        x_new.append(extract_histogram(cv2.imread(image)))
    x_new = np.array(x_new)

    y_new = clf.predict(x_new)
    #TODO тут тоже нужно изменить на то которое указано в задании
    print("Predicted class for cat.1016.jpg: ", y_new[16])
    print("Predicted class for cat.1024.jpg: ", y_new[24])
    print("Predicted class for cat.1006.jpg: ", y_new[56])
    print("Predicted class for cat.1033.jpg: ", y_new[83])


if __name__ == '__main__':
    main()