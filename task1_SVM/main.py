import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    # импорт данных
    dataset = pd.read_csv('54_25.csv', header=None)
    # центрирование
    dataset = dataset - dataset.mean(axis=0)

    # инициализация МГК
    pca_test = PCA(n_components=2, svd_solver='full')
    pca_test.fit(dataset)
    dataset2 = pca_test.transform(dataset)
    print("New coordinates of the first object: ", dataset2[0])

    # визуализация МГК
    for i in range(60):
        plt.scatter(dataset2[i][0], dataset2[i][1])
    plt.show()
    plt.clf()

    # подсчет объясненной дисперсии
    explained_variance = pca_test.explained_variance_ratio_
    sum = 0
    for i in range(len(explained_variance)):
        sum += explained_variance[i]
    print("Explained variance:", sum)

    # Построение графика объясненной дисперсии для 10 ГК
    pca_test = PCA(n_components=10, svd_solver='full')
    pca_test.fit(dataset)
    explained_variance = pca_test.explained_variance_ratio_
    for i in range(1, len(explained_variance)):
        explained_variance[i] += explained_variance[i - 1]
    plt.plot(np.arange(1, 11), explained_variance)
    plt.show()
    plt.clf()

    # восстановление логотипа. Умножаем матрицу счета на матрицу весов транспонироваанную
    z = pd.read_csv('X_reduced_536.csv', header=None, sep=';')
    fi = pd.read_csv('X_loadings_536.csv', header=None, sep=';')
    fi = np.array(fi).transpose()
    tr = np.dot(z, fi)

    # отрисовка логотипа
    result = pd.DataFrame(tr)
    for i in range(100):
        for j in range(100):
            if result[i][j] <= 0:
                plt.scatter(i, -j)
    plt.imshow(tr)
    plt.show()


if __name__ == '__main__':
    main()
