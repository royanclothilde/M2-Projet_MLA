# -*- coding: utf-8 -*-
"""
@author: clothilde royan
"""

import pandas as pd
import numpy as np
import glob
import cv2

nb_image = 200
dir_covid = "train/COVID/"
dir_sain = "train/NORMAL/"
dir_autre = "train/PNEUMONIA_virus/"

datas = pd.DataFrame(columns=["filepath", "labels"])
j = 0

# Images poumons sains
filenames = glob.glob(dir_sain + "*.*")
nb = 0

for f in filenames:
    if nb <= nb_image:
        datas.loc[j] = [f, 0]
        j += 1
    nb += 1
print("Sains = ", len(datas))

# Images poumons COVID
filenames = glob.glob(dir_covid + "*.*")
nb = 0

for f in filenames:
    if nb <= nb_image:
        datas.loc[j] = [f, 1]
        j += 1
    nb += 1
print("COVID = ", len(datas))
# Images poumons autres
filenames = glob.glob(dir_autre + "*.*")
nb = 0

for f in filenames:
    if nb <= nb_image:
        datas.loc[j] = [f, 2]
        j += 1
    nb += 1
print("Autres = ", len(datas))

# On mélange les lignes au hasard
datas = datas.sample(frac=1).reset_index(drop=True)
datas = datas.sample(frac=1).reset_index(drop=True)

# On sélectionne 70% des données pour l'apprentissage
nb_train = round((70/100)*len(datas))

# On créer les différentes variables pour les diférentes bases
basetrain = []
basetest = []
labeltrain = []
labeltest = []

for i in range(len(datas)):
    if i <= nb_train:
        # On ouvre le fichier
        pic = cv2.imread(datas["filepath"][i], 0)
        

        # On le redimensionne
        pic = cv2.resize(pic, (300, 300))

        
        basetrain.append(pic)
        labeltrain.append(datas["labels"][i])

    else:
        # On ouvre le fichier
        pic = cv2.imread(datas["filepath"][i], 0)

        # On le redimensionne
        pic = cv2.resize(pic, (300, 300))

        basetest.append(pic)
        labeltest.append(datas["labels"][i])

np.save("Test/BaseTrain.npy", np.array(basetrain))
np.save("Test/BaseTest.npy", np.array(basetest))
np.save("Test/LabelTrain.npy", np.array(labeltrain))
np.save("Test/LabelTest.npy", np.array(labeltest))
