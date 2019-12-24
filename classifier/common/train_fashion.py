import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# print(train_images.shape)

# defining class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plot image
# train_images = train_images / 255.0
# test_images = test_images / 255.0
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# make model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# train model
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

# test accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Accuracy: ', test_acc)


predictions = model.predict(test_images)
pred_results = np.argmax(predictions, axis=1)
# print(pred_results.shape)


# print a sample
def plot_image(i, pred_arr, true_labels, imgs):
    """
    Plots single image with names, prediction, and true label displayed,
    extinguished by color blue: true, red: wrong
    """
    pred, true_label, img = pred_arr[i], true_labels[i], imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    print('pred: ', pred, 'true label: ', true_label)
    color = 'blue' if pred == true_label else 'red'
    plt.xlabel("{} {} {}".format(class_names[pred],
                                 pred,
                                 class_names[true_label]),
               color=color)


# plot an image
plt.figure(figsize=(10, 10))
for i in range(0, 25):
    plt.subplot(5, 5, i + 1)
    plot_image(i, pred_results, test_labels, test_images)
plt.show()
