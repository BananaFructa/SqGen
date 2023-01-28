import tensorflow as tf
import numpy as np
import copy
import random

def genInvInput(vecs, invIndex, low, high):
    formatted = []
    for l in range(0,len(vecs)):
        for i in range(0,11):
            f = low + (i/10) * (high - low)
            vecs[l][invIndex] = f
            formatted.append(copy.deepcopy(vecs[l]))
    return formatted

def duplicateArray(c,arr):
    dup = []
    for i in range(0,c):
        dup.append(arr)
    return dup

def sqgenSaveModel(path,model):
    i = 0
    for layer in model.layers:
        np.save(path + "/" + str(i) + ".npy",np.asfortranarray(layer.get_weights()[0]))
        i += 1
        biases = layer.get_weights()[1]
        biases = np.reshape(biases,(1,biases.shape[0]))
        np.save(path + "/" + str(i) + ".npy",np.asfortranarray(biases))
        i += 1

T_REP = 2/5

EMPTY = genInvInput(genInvInput([[0,0,   0,0,0,0,    0,0,0,0]],0,0,1),1,0,1)

#for i in range(1,10):
#    FOOD_VEC = genInvInput(FOOD_VEC,i,0,1)
FOOD_LABELS_VEC = []
for i in range(0,len(FOOD_VEC)):
    FOOD_LABELS_VEC += [[FOOD_VEC[i][1]]]

MUL_VEC  = genInvInput(genInvInput([[0,0,    0,0,0,0,     0,0,0,0]],0,T_REP,1),1,0,1)
MUL_LABELS_VEC = []
for i in range(0,len(MUL_VEC)):
    MUL_LABELS_VEC += [[-1]]

DATA = np.array(FOOD_VEC + MUL_VEC)
LABELS = np.array(FOOD_LABELS_VEC + MUL_LABELS_VEC)

def buildModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10,activation="tanh",input_shape=(10,)))
    model.add(tf.keras.layers.Dense(10,activation="tanh"))
    model.add(tf.keras.layers.Dense(1,activation="tanh"))
    return model

def trainBSG():
    model = buildModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    lastLoss = 1

    while lastLoss > 0.0001:
        for i in range(0,DATA.shape[0]):
            for j in range(2,10):
                DATA[i][j] = random.uniform(-1,1)
        with tf.GradientTape() as tape:
            prediction = model(DATA)
            loss = tf.keras.losses.MSE(LABELS,prediction)
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            
            lastLoss = np.mean(loss)

            print("Loss - " + str(lastLoss))

    sqgenSaveModel("./B_SG_NPY",model)
    model.save("B_SG")

trainBSG()