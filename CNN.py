
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Fonction pour l'affichage des courbes
def affiche(history):

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# Fonction pour le CNN
def CNN(x_train, y_train, x_test, y_test, nb_label):
      
      # Normalisation
      #x_train = x_train / 255.0
      #x_test = x_test / 255.0

      # Reduction des images pour éviter 
      x = x_train[0].shape[0]
      y = x_train[0].shape[1]
      cropx = 50
      cropy = 50
      lx = x-2*cropx
      ly = y-2*cropy
      nx = x-cropx
      ny = y-cropy
      startx = x//2-(cropx//2)
      starty = y//2-(cropy//2)
      x_train_c = np.zeros((len(x_train), lx, ly))
      x_test_c = np.zeros((len(x_test), lx, ly))
      for i in range(len(x_train)):
          x_train_c[i] = x_train[i, cropx:nx, cropy:ny]
      for i in range(len(x_test)):
          x_test_c[i] = x_test[i, cropx:nx, cropy:ny]

      # POUR LES CNN : On rajoute une dimension pour spécifier qu'il s'agit d'imgages en NdG
      x_train = x_train_c.reshape(x_train_c.shape[0], x_train_c.shape[1], x_train_c.shape[2], 1)
      print(x_train.shape)
      x_test = x_test_c.reshape(x_test_c.shape[0], x_test_c.shape[1], x_test_c.shape[2], 1)

      # One hot encoding
      y_train = tf.keras.utils.to_categorical(y_train)
      y_test = tf.keras.utils.to_categorical(y_test)

      filter_size_conv1 = (5, 5)

      # Définition de l'architecture
      model = tf.keras.models.Sequential()
      model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=filter_size_conv1, padding="same", activation='relu',
                                      input_shape=(200, 200, 1)))
      model.add(tf.keras.layers.AveragePooling2D())
      model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid", activation='relu'))
      model.add(tf.keras.layers.AveragePooling2D())
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dense(120, activation='relu'))
      model.add(tf.keras.layers.Dense(84, activation='relu'))
      model.add(tf.keras.layers.Dense(nb_label, activation='softmax'))

      print(model.summary())

      # Optimisation des paramètres
      sgd = tf.keras.optimizers.Adam()
      model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

      # Entrainement
      history = model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))
      test_loss, test_acc = model.evaluate(x_test, y_test)

      # Evaluation
      y_pred = model.predict(x_test)
      y_pred = y_pred.argmax(axis=-1)
      y_test = np.argmax(y_test, axis=1)
      
      print('Test accuracy :', test_acc)

      df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
      print("Confusion Matrix :")
      print(df_confusion)
      affiche(history)


# Main 
# Avec 2 labels
# Chargement des bases d’apprentissage et de test :
x_train = np.load('Test_2labels/BaseTrain_2.npy')
y_train = np.load('Test_2labels/LabelTrain_2.npy')

x_test = np.load('Test_2labels/BaseTest_2.npy')
y_test = np.load('Test_2labels/LabelTest_2.npy')

CNN(x_train, y_train, x_test, y_test, 2)


# Avec 3 labels
# Chargement des bases d’apprentissage et de test:
x_train = np.load('Test_3labels/BaseTrain_3.npy')
y_train = np.load('Test_3labels/LabelTrain_3.npy')

x_test = np.load('Test_3labels/BaseTest_3.npy')
y_test = np.load('Test_3labels/LabelTest_3.npy')

CNN(x_train, y_train, x_test, y_test, 3)