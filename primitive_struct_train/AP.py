import tensorflow as tf
import numpy as np
import os
import copy

T_EAT = 0.1
T_SHARE = 2/5
T_REP = 2/5

SIGN_COUNT = 5

# A1 -> 1
# A2 -> 0.5
# B1 -> -1
# B2 -> -0.5

# Current F_i
# Current F_t
# Visual input L R U D
# Signal input L R U D

def genInvInput(vecs, invIndex, low, high):
    formatted = []
    for l in range(0,len(vecs)):
        for i in range(0,11):
            f = low + (i/10) * (high-low)
            vecs[l][invIndex] = f
            formatted.append(copy.deepcopy(vecs[l]))
    return formatted

def genInvInputVec(vecs, invIndex, vecVar):
    formatted = []
    for l in range(0,len(vecs)):
        for i in range(0,len(vecVar)):
            vecs[l][invIndex] = vecVar[i]
            formatted.append(copy.deepcopy(vecs[l]))
    return formatted

def duplicateArray(c,arr):
    dup = []
    for i in range(0,c):
        dup.append(arr)
    return dup

def joinArr(arr):
    joined = []
    for i in range(0,len(arr)):
        joined += arr[i]
    return joined

def processLabels(labels,inputs):
    l = []
    for i in range(0,len(labels)):
        l += duplicateArray(len(inputs[i]),labels[i])
    return l

def softmax(x):
    return (np.exp(x)/np.exp(x).sum())

A_CLOSE = 0.997847
A_FAR   = 0.5
B_CLOSE = -0.998465
B_FAR   = -0.5
A_CLOSE_B_FAR = 0.750012
B_CLOSE_A_FAR = -0.750009

BORDER_MARGIN = 0.05

POS_VIEW_VEC = [A_CLOSE,B_CLOSE,A_CLOSE_B_FAR,B_CLOSE_A_FAR]

EAT_VEC     = [1,   0,  0,  0,  0,  0,  0,  0,  0]
MUL_VEC     = [0,   1,  0,  0,  0,  0,  0,  0,  0]
UP_VEC      = [0,   0,  1,  0,  0,  0,  0,  0,  0]
DOWN_VEC    = [0,   0,  0,  1,  0,  0,  0,  0,  0]
RIGHT_VEC   = [0,   0,  0,  0,  1,  0,  0,  0,  0]
LEFT_VEC    = [0,   0,  0,  0,  0,  1,  0,  0,  0]
ATTACK_VEC  = [0,   0,  0,  0,  0,  0,  1,  0,  0]
SHARE_VEC   = [0,   0,  0,  0,  0,  0,  0,  1,  0]
SIGNAL_VEC  = [0,   0,  0,  0,  0,  0,  0,  0,  1]

#                                      | |
A_1      = genInvInput(genInvInput(  [[0,0,      0,A_CLOSE,0,B_FAR,           0,0,0,-1]]  ,0,0,1),1,0,1) # left

#                                      | |
A_2      = genInvInput(genInvInput(  [[0,0,      A_CLOSE_B_FAR,0,0,B_FAR,           0,0,0,-1]]  ,0,0,1),1,0,1) # right

#                                                  | |                            |
A_3_1    = genInvInput(genInvInput(genInvInput(  [[0,0,     0,0,0,B_CLOSE,  0,0,0,0]]  ,0,0,1),1,0,1),9,-1,1) # change sig

#                                                              | |                              |     |
A_3_2    = genInvInput(genInvInput(genInvInput(genInvInput(  [[0,0,     B_FAR,0,0,B_FAR,        0,0,0,0]]    ,0,0,1),1,0,1),6,-1,1),9,-1,1) # change sig

#                                                              | |                                |   |
A_3_3    = genInvInput(genInvInput(genInvInput(genInvInput(  [[0,0,      0,B_FAR,0,B_FAR,       0,0,0,0]]  ,0,0,1),1,0,1),7,-1,1),9,0,1) # change sig

#                                       | |
A_4      = genInvInput(genInvInput(   [[0,0,    0,0,0,B_FAR,    0,0,0,-1]]   ,0,0,1),1,0,1) # left

#                                       | |
A_5      = genInvInput(genInvInput(   [[0,0,       0,B_FAR,0,B_FAR,        0,-1,0,-1]]  ,0,0,1),1,0,1) # multiply

#                                                                           | |          V                   |   |
A_6      = genInvInput(genInvInput(genInvInputVec(genInvInput(genInvInput([[0,0,       0,0,0,B_FAR,        0,0,0,0]],0,0,1),1,0,1),3,POS_VIEW_VEC),7,-1,1),9,0,1) # left

#                                                                           | |        V                   |     |
A_7      = genInvInput(genInvInput(genInvInputVec(genInvInput(genInvInput([[0,0,       0,0,0,B_FAR,        0,0,0,0]],0,0,1),1,0,1),2,POS_VIEW_VEC),6,-1,1),9,0,1) # right

#                                                                           | |            V                   | |
A_8      = genInvInput(genInvInput(genInvInputVec(genInvInput(genInvInput([[0,0,       0,0,0,B_FAR,        0,0,0,0]],0,0,1),1,0,1),4,POS_VIEW_VEC),8,-1,1),9,0,1) # down

#                                                V V                                  V             F_i > T_share ; F_t inv.
A_9      = genInvInput(genInvInput(genInvInput([[0,0,       0,0,0,B_FAR,        0,0,0,0]],0,T_SHARE,1),1,0,1),9,0,1) # share

#                                                V V                                  V             F_i < T_share ; F_t > T_eat
A_10     = genInvInput(genInvInput(genInvInput([[0,0,       0,0,0,B_FAR,        0,0,0,0]],0,0,T_SHARE-BORDER_MARGIN),1,T_EAT,1),9,0,1) # eat

#                                                V V                                  V             F_i < T_share ; F_t < T_eat
A_11     = genInvInput(genInvInput(genInvInput([[0,0,       0,0,0,B_FAR,        0,0,0,0]],0,0,T_SHARE-BORDER_MARGIN),1,0,T_EAT-BORDER_MARGIN),9,0,1)  # respone in relation to D sig

# ATTENCTION "A" DATA
A_DATA_VEC = [A_1, A_2 , A_3_1, A_3_2 , A_3_3 , A_4 , A_5 , A_6 , A_7 , A_8, A_9 ,A_10, A_11]
A_DATA = np.array(joinArr(A_DATA_VEC))
# FINAL LABELS
A_LABELS_VEC = [LEFT_VEC,RIGHT_VEC,SIGNAL_VEC,SIGNAL_VEC,SIGNAL_VEC,LEFT_VEC,MUL_VEC,LEFT_VEC,RIGHT_VEC,DOWN_VEC,SHARE_VEC,EAT_VEC] # last is omitted as it depends on the case
A_LABELS_VEC_PROCESSED = processLabels(A_LABELS_VEC,A_DATA_VEC)

for i in range(0,len(A_11)):
    bSig = A_11[i][9]
    action = [0,0,0,0,0,0,0,0,0]
    prob = softmax([0,bSig,0,0])
    for j in range(0,4):
        action[2 + j] = prob[j]
    A_LABELS_VEC_PROCESSED.append(action)
A_LABELS = np.array(A_LABELS_VEC_PROCESSED)

#                                     | |                                         F_i > T_rep ; F_t inv.
B_1      = genInvInput(genInvInput( [[0,0,        0,A_FAR,A_CLOSE,0,      0,0,0,0]] ,0,T_REP,1),1,0,1) # multiply
#                                     | |                                         F_i < T_rep ; F_t inv.
B_1_1    = genInvInput(genInvInput( [[0,0,        0,A_FAR,A_CLOSE,0,      0,0,0,0]] ,0,0,T_EAT-BORDER_MARGIN),1,0,1) # change signal
#                                                 | |                                              |
B_2      = genInvInput(genInvInput(genInvInput( [[0,0,       A_FAR,B_CLOSE_A_FAR,A_CLOSE,0,      0,0,0,0]]  ,0,0,1),1,0,1),7,-1,1) # change sig
#                                                | |                                            |
B_3      = genInvInput(genInvInput(genInvInput([[0,0,       B_CLOSE_A_FAR,A_FAR,A_CLOSE,0,      0,0,0,0]],0,0,1),1,0,1),6,-1,1) # right
#                                                                                 | |        | |       |
B_4      = genInvInputVec(genInvInputVec(genInvInputVec(genInvInput(genInvInput([[0,0,       0,0,A_FAR,0,        0,0,0,0]],0,0,1),1,0,1),2,[-1,1]),3,[-1,1]),5,[-1,1]) # attack
#                                    | |
B_5      = genInvInputVec(genInvInput(genInvInput([[0,0,        A_FAR,0,A_FAR,0,        0,0,0,0]],0,0,1),1,0,1),3,[0,-0.75,0.5,-0.5,0.75]) # left
#                                    | |
B_6      = genInvInputVec(genInvInput(genInvInput([[0,0,        0,A_FAR,A_FAR,0,        0,0,0,0]],0,0,1),1,0,1),2,[0,-0.75,-0.5,0.75]) # right
#                                    | |
B_7      = genInvInput(genInvInput([[0,0,       0,0,A_CLOSE,0,       0,0,0,0]],0,0,1),1,0,1) # down
#                                    | |
B_8      = genInvInput(genInvInput([[0,0,       0,0,0,0,        0,0,0,0]],0,0,1),1,0,1) # up
#                                    | |
B_9      = genInvInput(genInvInput([[0,0,       0,0,A_FAR,0,        0,0,0,0]],0,0,1),1,0,1) # change sig

B_DATA_VEC = [B_1,B_1_1,B_2,B_3,B_4,B_5,B_6,B_7,B_8,B_9]
B_LABEL_VEC = [MUL_VEC,SIGNAL_VEC,SIGNAL_VEC,RIGHT_VEC,ATTACK_VEC,LEFT_VEC,RIGHT_VEC,DOWN_VEC,UP_VEC,SIGNAL_VEC]

B_DATA = np.array(joinArr(B_DATA_VEC))
B_LABELS = np.array(processLabels(B_LABEL_VEC,B_DATA_VEC))

def buildAPModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(20,activation="tanh",input_shape=(10,)))
    model.add(tf.keras.layers.Dense(20,activation="tanh"))
    model.add(tf.keras.layers.Dense(20,activation="tanh"))
    model.add(tf.keras.layers.Dense(9,activation="softmax"))
    return model

def getOptimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.01)

def sqgenSaveModel(path,model):
    i = 0
    for layer in model.layers:
        np.save(path + "/" + str(i) + ".npy",np.asfortranarray(layer.get_weights()[0]))
        i += 1
        biases = layer.get_weights()[1]
        biases = np.reshape(biases,(1,biases.shape[0]))
        np.save(path + "/" + str(i) + ".npy",np.asfortranarray(biases))
        i += 1

def trainAAP():
    model = buildAPModel()
    optimizer = getOptimizer()
    lastLoss = 1
    while lastLoss > 0.000001:
        with tf.GradientTape() as tape:
            prediction = model(A_DATA)
            loss = tf.keras.losses.MSE(A_LABELS,prediction)
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            lastLoss = np.mean(loss)
            print("Loss - " + str(lastLoss))
    sqgenSaveModel("./A_AP_NPY",model)
    model.save("A_AP")

def trainBAP():
    model = buildAPModel()
    optimizer = getOptimizer()
    lastLoss = 1
    while lastLoss > 0.000001:
        with tf.GradientTape() as tape:
            prediction = model(B_DATA)
            loss = tf.keras.losses.MSE(B_LABELS,prediction)
            gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            lastLoss = np.mean(loss)
            print("Loss - " + str(lastLoss))
    sqgenSaveModel("./B_AP_NPY",model)
    model.save("B_AP")

np.set_printoptions(suppress=True)
#trainAAP()
trainBAP()