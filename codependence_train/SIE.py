import tensorflow as tf
import numpy as np
import os

# A Identifier 1 1 0 0 0   ==SIE==>  1 (At 1 tiles) /  0.5 (At 2 tiles)
# B Identifier 0 0 0 1 1   ==SIE==> -1 (At 1 tiles) / -0.5 (At 2 tiles)

NULL = np.array([0,0,0,0,0])
A_IDENTIFIER_1 = np.array([1,1,0,0,0])#.reshape((1,5,1))
A_IDENTIFIER_2 = np.array([0.5,0.5,0,0,0])#.reshape((1,5,1))
B_IDENTIFIER_1 = np.array([0,0,0,1,1])#.reshape((1,5,1))
B_IDENTIFIER_2 = np.array([0,0,0,0.5,0.5])#.reshape((1,5,1))

NULL_RES = np.array([0])
A_SIE_RES_1 = np.array([1])
A_SIE_RES_2 = np.array([0.5])
B_SIE_RES_1 = np.array([-1])
B_SIE_RES_2 = np.array([-0.5])
AB12 = np.array([0.75])
AB21 = np.array([-0.75])

DATA = [NULL,A_IDENTIFIER_1,A_IDENTIFIER_2,B_IDENTIFIER_1,B_IDENTIFIER_2,np.add(A_IDENTIFIER_1,B_IDENTIFIER_2),np.add(A_IDENTIFIER_2,B_IDENTIFIER_1)]
LABELS = [NULL_RES,A_SIE_RES_1,A_SIE_RES_2,B_SIE_RES_1,B_SIE_RES_2,AB12,AB21]

NP_DATA = np.array(DATA)
NP_LABELS = np.array(LABELS)

def buildSIE():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5,input_shape = (5,),activation="tanh"))
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

    while last_loss > 0.000001:
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