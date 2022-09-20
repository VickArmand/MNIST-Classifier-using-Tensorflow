import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
# from mnist.mnistrestructured.deeplearningmodels import MyCustomModel, functional_model
from myutils import displaysampleimages
#from tensorflow.python.layers import activations
# Approaches to use in building neural networks
# Sequential- tensorflow.keras.Sequential
# Functional approach- build a function that returns a model
# inherit the base class- tensorflow.keras.Model
model=tensorflow.keras.Sequential(
    [
    # layer 1
        Input(shape=(28,28,1)),#here one sets the shape that corresponds to the shape of the dataset
        Conv2D(32,(3,3),activation='relu'), #here we are creating 32 filters of 3 by 3 each
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
    # layer 2
        Conv2D(128,(3,3),activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
    # layer 3
        GlobalAvgPool2D(),# finds average of output values from BatchNormalization()
        Dense(64,activation='relu'),
        Dense(10,activation='softmax'),#output layer
        # using softmax in the output layer yields 10 probabilities whereby the highest probability yields the correct outcome
    ]
)


if __name__=='__main__':
    (x_train,y_train),(x_test,y_test)=tensorflow.keras.datasets.mnist.load_data()
    # print("x_train's shape is",x_train.shape)
    # print("y_train's shape is",y_train.shape)
    # print("x_test's shape is",x_test.shape)
    # print("y_test's shape is",y_test.shape)
    if False:
        # Function to display 25 images and labels in the dataset
        displaysampleimages(x_train,y_train)
    # Normalizing data
    x_train=x_train.astype('float32')/255
    x_test=x_test.astype('float32')/255
    x_train=np.expand_dims(x_train,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)
    # print("x_train's shape is",x_train.shape)
    # print("y_train's shape is",y_train.shape)
    # print("x_test's shape is",x_test.shape)
    # print("y_test's shape is",y_test.shape)

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
    # categorical_crossentropy loss function works when the labels on y_train and y_test are one hot encoded example label 2 one hot encodng [0,1,0,0,0,0,0,0,0,0] otherwise you can use sparse_categorical_crossentropy
    # model training
    model.fit(x_train,y_train,batch_size=64,epochs=3,validation_split=0.2)#batch_size represents the number of images to be seen by the model every time,epochs represents the number of times the model sees the datasets 
    # here 80% wil be for training and 20% will be for validation
# If you want to build a solid model you have to follow that specific protocol of splitting your data into three sets:
#  One for training, one for validation and one for final evaluation, which is the test set.
# The idea is that you train on your training data and tune your model with the results of metrics (accuracy, loss etc) that you get from your validation set.
# Train-Set: The data-set on which the model is being trained on. This is the only data-set on which the weights are updated during back-propagation.
# Validation-Set (Development Set): The data-set on which we want our model to perform well. During the training process we tune hyper-parameters such that the model performs well on dev-set (but don't use dev-set for training, it is only used to see the performance such that we can decide on how to change the hyper-parameters and after changing hyper-parameters we continue our training on train-set). Dev-set is only used for tuning hyper-parameters to make the model eligible for working well on unknown data (here dev-set is considered as a representative of unknown data-set as it is not directly used for training and additionally saying the hyper-parameters are like tuning knobs to change the way of training) and no back-propagation occurs on dev-set and hence no direct learning from it.
# Test-Set: We just use it for unbiased estimation. Like dev-set, no training occurs on test-set. The only difference from validation-set (dev- set) is that we don't even tune the hyper-parameters here and just see how well our model have learnt to generalize. Although, like test-set, dev-set is not directly used for training, but as we repeatedly tune hyper-parameters targeting the dev-set, our model indirectly learns the patterns from dev-set and the dev-set becomes no longer unknown to the model. Hence we need another fresh copy of dev-set which is not even used for hyper parameter tuning, and we call this fresh copy of dev-set as test set. As according to the definition of test-set it should be "unknown" to the model. But if we cannot manage a fresh and unseen test set like this, then sometimes we say the dev-set as the test-set.
# The test set is used for model evaluation in order to know how the model performs on unseen images

# model evaluation on test set
    model.evaluate(x_test,y_test, batch_size=64)
# if you want to convert your labels to support categorical_crossentropy 
    # y_train=tensorflow.keras.utils.to_categorical(y_train,10)
    # y_test=tensorflow.keras.utils.to_categorical(y_test,10)