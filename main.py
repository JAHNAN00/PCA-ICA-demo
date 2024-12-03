import numpy as np
import matplotlib.pyplot as plt
import struct
from PIL import Image
from sklearn.decomposition import FastICA
from scipy.special import softmax
from scipy.linalg import eigh

from src.ica import ICA
from src.pca import PCA

def read_img(bmp_path):
    img = Image.open(bmp_path)
    np_array = np.array(img)
    return np_array

def whiten(X, thre=1e-10, truth=3):
    n_samples = X.shape[1]
    mean = np.mean(X, axis=1)
    mean = np.reshape(mean, (len(mean), 1))
    ones = np.ones((1, n_samples))
    meanMatrix = np.dot(mean, ones)
    X = X - meanMatrix
    covMatrix = np.cov(X) 
    U, d, V = np.linalg.svd(covMatrix, full_matrices=False)
    V = V.T
    d = d[:truth]  
    D = np.diag(d)  
    E = U[:, :truth] 
    whiteningMatrix = np.dot(np.linalg.inv(np.sqrt(D)), E.T)  
    whitenedMatrix = np.dot(whiteningMatrix, X) 
    return whitenedMatrix

def recover(ss, ys):
    n, D = ss.shape
    coef_mat = np.zeros((n, n))
    outs = np.zeros_like(ys)
    for i in range(n):
        for j in range(n):
            coef_mat[i, j] = np.corrcoef(ys[i], ss[j])[0, 1]
    coef_mat_abs = np.abs(coef_mat)
    labels = np.argmax(coef_mat_abs, axis=1)
    for i in range(n):
        y = ys[i]
        j = labels[i]
        s = ss[j]
        mu = np.mean(s)
        s0 = s - mu
        alpha = np.sum(s0 * y) / np.sum(s0 * s0)
        outs[i] = y / alpha + mu
        outs[outs < 0.0] = 0.0
        outs[outs > 1.0] = 1.0

    return outs

def mix_pic(imgs, n, m):
    return np.matmul(softmax(np.random.randn(m, n), axis=1), imgs[0:n])

def task_ica():
    np.random.seed(0)

    imgs = []
    for i in range(4):
        img_path ='./dataset/scene/{}.bmp'.format(i+1)
        img = (np.array(Image.open(img_path)) / 255).flatten()
        imgs.append(img)
    imgs = np.array(imgs)

    imgs_mixed = mix_pic(imgs, m=4, n=4)
    imgs_mixed = whiten(imgs_mixed, truth=4)

    ica = FastICA(n_components=4)
    #ica = ICA(n_components=4)
    img_separated = ica.fit_transform(imgs_mixed.T)
    print(img_separated.shape)
    imgs_recover = recover(imgs[0:4], img_separated.T)
    for i in range(4):
        plt.imsave('./ica{}.png'.format(i+1), imgs_recover[i].reshape(256, 512), cmap='gray')
        
def task_pca():
    np.random.seed(0)
    faces = []
    for i in range(40):
        face=[]
        for j in range(10):
            img_path ='./dataset/face/s{}/{}.bmp'.format(i+1,j+1)
            img = (np.array(Image.open(img_path)) / 255).flatten()

            face.append(img)
        faces.append(face)
    faces = np.array(faces)
    print(f"before:{faces.shape}")

    pca = PCA()
    outputs=[]
    for face in faces:
        mean, eigen_vector, output=pca.fit(faces[0])
        outputs.append(output)
    outputs = np.array(outputs)
    print(f"After:{outputs.shape}")
    

if __name__ == "__main__":
    #task_pca()
    task_ica()