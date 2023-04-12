import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
import time

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# plot images
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(classes[train_labels[i][0]])
plt.show()

# reshape images for ANN input
train_images = train_images.reshape((train_images.shape[0], 32*32*3))
test_images = test_images.reshape((test_images.shape[0], 32*32*3))

# model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(32*32*3,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training
start_time = time.time()
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# plot history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, .5])
plt.legend(loc='lower right')

print("--- %s seconds ---" % (time.time() - start_time))
