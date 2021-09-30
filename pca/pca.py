import numpy as np


def zeromean(datamat):
    meanval = np.mean(datamat, axis=0)
    newdata = datamat - meanval
    return newdata


def percentage2n(datamat, percentage):
    newdata = zeromean(datamat)
    covmat = np.cov(newdata, rowvar=0)
    eigvals, eigvects = np.linalg.eig(np.mat(covmat))
    sortarray = np.sort(eigvals)
    sortarray = sortarray[::-1]
    arraysum = sum(sortarray)
    tmpsum = 0
    num = 0
    for i in sortarray:
        tmpsum += i
        num += 1
        if tmpsum >= arraysum * percentage:
            return num


def n2percentage(datamat, n):
    newdata = zeromean(datamat)
    covmat = np.cov(newdata, rowvar=0)
    eigvals, eigvects = np.linalg.eig(np.mat(covmat))
    sortarray_n_sum = sum(np.sort(eigvals)[-1:-(n + 1):-1])
    sortarray_sum = sum(np.sort(eigvals)[::-1])
    percentage = sortarray_n_sum / sortarray_sum
    return percentage


def pca(datamat, percentage=0.99, n=None):
    newdata = zeromean(datamat)
    covmat = np.cov(newdata, rowvar=0)
    eigvals, eigvects = np.linalg.eig(np.mat(covmat))
    if n is None:
        n = percentage2n(datamat, percentage)  
    eigvalindice = np.argsort(eigvals)
    n_eigvalindice = eigvalindice[-1:-(n + 1):-1]
    n_eigvect = eigvects[:, n_eigvalindice]
    lowddatamat = newdata * n_eigvect
    return lowddatamat


if __name__ == '__main__':
    data = np.mat([[4, 2, 3], [4, 5, 6], [1, 8, 9], [10, 11, 12], [13, 14, 15]])
    print(pca(data))
    print(n2percentage(data, 1))
