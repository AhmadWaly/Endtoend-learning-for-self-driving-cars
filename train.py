from keras.models import *
from keras.callbacks import *
import keras.backend as K
from model import *
import cv2
from data_pre import*
import matplotlib.pyplot as plt

print('hi training')
# train_X,train_y,test_X,test_y=return_data()
def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)

def atan_layer_shape(input_shape):
    return input_shape
def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
#
model=getmodel()

# model=load_model('model_60.h5',compile=False,custom_objects={'tf':tf,'atan_layer':atan_layer,'atan_layer_shape':atan_layer_shape,"soft_acc":soft_acc})

model.compile(optimizer=Adam(epsilon=1e-08,lr=0.001), loss='mse',metrics=[soft_acc])
# # model.fit(x=train_X, y=train_y, batch_size=100, epochs=30,validation_split=0.2)
#
val_X,val_y=get_val_data(X,y)
val_set=(val_X,val_y)
history=model.fit_generator(generate_arrays_from_file(trainfile), steps_per_epoch=625, epochs=30,verbose=1,validation_data=val_set)

model.save('model_pretrained_credit_ta7t.h5')


training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();
#############################################################################################################################################################################
#############################################################################################################################################################################

training_acc = history.history['acc']
test_acc = history.history['val_acc']

# Create count of the number of epochs
epoch_count = range(1, len(training_acc) + 1)

# Visualize loss history
plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training acc', 'Test acc'])
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.show();

#############################################################################################################################################################################
#############################################################################################################################################################################

# model=getmodel()
# model=load_model('model.h5',compile=True,custom_objects={'tf':tf , 'soft_acc':soft_acc,'atan_layer':atan_layer})

# model.compile(optimizer=Adam(epsilon=1e-08,lr=0.001), loss='mse',metrics=[soft_acc])
test_X,test_y=get_test_data(X,y)

loss,acc=model.evaluate(test_X, test_y, batch_size=50)
print('loss='+loss)
print('acc='+acc)
