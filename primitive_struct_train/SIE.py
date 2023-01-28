import tensorflow as tf
import numpy as np
import os

# A Identifier 1 1 0 0 0   ==SIE==>  1 (At 1 tiles) /  0.5 (At 2 tiles)
# B Identifier 0 0 0 1 1   ==SIE==> -1 (At 1 tiles) / -0.5 (At 2 tiles)
IDENTIFIER_ACTIVE   = np.array([1,0,0,0,0,    1,0,0,0,0])
IDENTIFIER_EMPTY    = np.array([0,0,0,0,0,    1,0,0,0,0])

ACTIVE = np.array([1])
EMPTY  = np.array([0])

DATA = [IDENTIFIER_EMPTY,IDENTIFIER_ACTIVE]
LABELS = [EMPTY,ACTIVE]

NP_DATA = np.array(DATA)
NP_LABELS = np.array(LABELS)

def buildSIE():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5,input_shape = (10,),activation="tanh"))
    model.add(tf.keras.layers.Dense(5,activation="tanh"))
    model.add(tf.keras.layers.Dense(5,activation="tanh"))
    model.add(tf.keras.layers.Dense(1,activation="tanh"))

    return model

def SIEOptimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.01)

def trainSIE():
    optimizer = SIEOptimizer()
    SIE = buildSIE()

    last_loss = 1

    while last_loss > 0.00001:
            with tf.GradientTape() as tape:
                prediction = SIE(NP_DATA)
                loss = tf.losses.MSE(NP_LABELS,prediction)
                gradients = tape.gradient(loss,SIE.trainable_variables)
                optimizer.apply_gradients(zip(gradients,SIE.trainable_variables))

                print("Loss - " + str(np.mean(loss)))
                print(prediction)

                last_loss = np.mean(loss)

    #os.mkdir("SIE")
    i = 0
    for layer in SIE.layers:
        np.save("./SIE/" + str(i) + ".npy",np.transpose(layer.get_weights()[0]).copy())
        i += 1
        biases = layer.get_weights()[1]
        biases = np.reshape(biases,(1,biases.shape[0]))
        np.save("./SIE/" + str(i) + ".npy",biases)
        i += 1

trainSIE()