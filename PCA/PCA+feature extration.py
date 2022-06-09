import numpy as np
import scipy
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt

from  sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.axes3d import Axes3D


def min_max_normalize(my_matrix):
    my_matrix_normalize = MinMaxScaler().fit_transform(my_matrix)
    return my_matrix_normalize


def standarization(my_matrix):
    my_matrix_standarize = StandardScaler().fit_transform(my_matrix)
    return my_matrix_standarize

df = pd.read_excel(r'G:\我的雲端硬碟\數據科學\期末報告\Date_Fruit_Datasets.xlsx')

trainset = df.iloc[0:896,0:34]
label = df.loc[:,'Class']
trainset = pd.DataFrame(standarization(trainset))

fig = plt.boxplot(trainset)
plt.show()


#trainset = pd.DataFrame(min_max_normalize(trainset))
#trainset = pd.DataFrame(standarization(trainset))


pca = PCA(n_components=10)

X_pca = pd.DataFrame(pca.fit_transform(trainset),columns=['PCA%i' %i for i in range(10)])
pca.explained_variance_ratio_
finalpca = pd.concat([X_pca,label],axis=1)

classes = [x for x in np.unique(np.array(label))]
colors = ['pink','red','grey','aquamarine','blue','purple','yellow']

finalpca.to_excel(r'G:\我的雲端硬碟\數據科學\期末報告\pca_data\pca_data_10.xlsx')

#plot PCA1 vs PCA2

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

for target , color in zip(classes,colors):

    indicestokeep = []

    for i,label in enumerate(finalpca['Class']):
        if label == target:
            indicestokeep.append(i)

    ax.scatter(finalpca.loc[indicestokeep,'PCA0']
              ,finalpca.loc[indicestokeep,'PCA1'])
ax.legend(classes,loc=1,fancybox=True, shadow=True, ncol=2, fontsize = 10)
plt.show()

#plot PCA1 vs PCA3

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC3')

for target , color in zip(classes,colors):

    indicestokeep = []

    for i,label in enumerate(finalpca['Class']):
        if label == target:
            indicestokeep.append(i)

    ax.scatter(finalpca.loc[indicestokeep,'PCA0']
              ,finalpca.loc[indicestokeep,'PCA2'])
ax.legend(classes,loc=1, fancybox=True, shadow=True, ncol=2, fontsize = 10)
plt.show()

#plot 3D PCA

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

for target , color in zip(classes,colors):

    indicestokeep = []

    for i,label in enumerate(finalpca['Class']):
        if label == target:
            indicestokeep.append(i)

    ax.scatter(finalpca.loc[indicestokeep,'PCA0']
              ,finalpca.loc[indicestokeep,'PCA1']
              ,finalpca.loc[indicestokeep,'PCA2']
              ,c = color , s = 30)
ax.legend(classes,loc=10, bbox_to_anchor=(0.06, 0.85),
          fancybox=True, shadow=True, ncol=2, fontsize = 10)
plt.show()


