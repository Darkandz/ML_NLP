import cv2
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from imutils import paths
from sklearn.metrics import accuracy_score
import numpy as np


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
            y.append(1)
        else:
            y.append(0)
        counter -= 1
    #TODO Изменить параметры модели
    clf1 = LinearSVC(C=1.74,
                     random_state=220)
    # TODO Изменить параметры модели
    clf2 = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',
                                                                   min_samples_leaf=10,
                                                                   max_leaf_nodes=20,
                                                                   random_state=220),
                             n_estimators=19,
                             random_state=220)
    # TODO Изменить параметры модели
    clf3 = RandomForestClassifier(n_estimators=19,
                                  criterion='entropy',
                                  min_samples_leaf=10,
                                  max_leaf_nodes=20,
                                  random_state=220)
    # TODO Изменить параметры модели
    clf4 = LogisticRegression(solver='lbfgs', random_state=220)

    estimators = [('svr', clf1),
                  ('bc', clf2),
                  ('rf', clf3)]

    clf = StackingClassifier(estimators=estimators, final_estimator=clf4, cv=2)
    clf.fit(x, y)

    x_new = []
    imagePaths2 = sorted(list(paths.list_images('test')))
    for image in imagePaths2:
        x_new.append(extract_histogram(cv2.imread(image)))
    x_new = np.array(x_new)

    y_pred = clf.predict(x)
    print("Accuracy score: ", accuracy_score(y, y_pred))

    y_new = clf.predict_proba(x_new)
    #TODO Необходимо изменить на фаши картинки число в скобочках
    print("Probability of classification image cat.1009.jpg as Class 1: ", y_new[9][1])
    print("Probability of classification image cat.1015.jpg as Class 1: ", y_new[15][1])
    print("Probability of classification image dog.1014.jpg as Class 1: ", y_new[64][1])
    print("Probability of classification image cat.1028.jpg as Class 1: ", y_new[28][1])


if __name__ == '__main__':
    main()