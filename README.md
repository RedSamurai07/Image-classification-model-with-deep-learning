# Image classification model with deep learning

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview

This project focuses on developing an automated image classification system using deep learning techniques. Utilizing the TensorFlow and Keras frameworks, the notebook outlines a complete pipeline for processing visual data, from initial preprocessing to model evaluation. The system is designed to handle multi-class classification, converting raw image input into meaningful categorical predictions.

### Executive Summary

The case study demonstrates the implementation of a neural network model aimed at accurately identifying objects within an image dataset. The methodology includes:

- Preprocessing: Implementation of Resizing and Rescaling layers to standardize image inputs for the model.

- Model Architecture: Construction of a Sequential model utilizing Flatten and Dense layers, along with regularizers to prevent overfitting.

- Reproducibility: Setting a specific random seed (111) to ensure consistent results across different training runs.

- Evaluation & Visualization: The project features a robust evaluation phase that randomly samples test images and compares predicted labels against true categories using a 5x3 visualization grid. This provides immediate visual confirmation of the model's accuracy, where correct predictions are typically highlighted in green and incorrect ones in red.
  
### Goal
The primary objectives for this case study are:

1. Develop a Scalable Classifier: Build a deep learning model capable of distinguishing between various categories within the dataset.

2. Optimize Model Performance: Use techniques such as rescaling and regularization to improve the model's ability to generalize to new, unseen images.

3. Establish Interpretability: Create visualization tools (like prediction plots) to analyze where the model succeeds and where it fails, allowing for targeted improvements.

4. Ensure Scientific Consistency: Implement rigorous data handling and seed-setting to allow for the validation and replication of findings.

### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1GK4tnY4_YfX8ccNhVtedEoGpsUUGOwzlysEVEQr_xpA/edit?gid=1748548740#gid=1748548740)

### Tools
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Data manipulation, Model development, fine tuning hyperparameters.
  
### Analysis
Python
Impprting all the libraries
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```
``` python
import os
import glob
import random
```
``` python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Rescaling, Resizing, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
tf.keras.utils.set_random_seed(111)
```

Data Loading
``` python
class_dirs = os.listdir("clothing-dataset-small/train")
image_dict = {} 
count_dict = {}

for cls in class_dirs:
    # Use glob to get all files inside the specific class folder
    file_paths = glob.glob(f'clothing-dataset-small/train/{cls}/*')
    
    if file_paths: # Check if folder isn't empty
        count_dict[cls] = len(file_paths)
        image_path = random.choice(file_paths)
        image_dict[cls] = tf.keras.utils.load_img(image_path)
        
plt.figure(figsize=(15, 12))
for i, (cls, img) in enumerate(image_dict.items()):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(f'{cls}\n{img.size}')
    plt.axis("off")
plt.show()
```
<img width="1133" height="987" alt="image" src="https://github.com/user-attachments/assets/01a7dfbb-a795-4595-8ce4-532df0036285" />

Dirtibution of training data across classes
``` python
## Plot of the Data Distribution of Training Data across Classes
df_count_train = pd.DataFrame({
    "class": count_dict.keys(),     
    "count": count_dict.values(),   
}).sort_values(by="count", ascending=False)

print("Count of training samples per class:\n", df_count_train)

df_count_train.plot.bar(
    x='class', 
    y='count', 
    title="Training Data Count per class",color="blue")
```
<img width="369" height="300" alt="image" src="https://github.com/user-attachments/assets/e70ac0f5-29ec-47a2-84b8-4ebeea5ddfa5" />

<img width="552" height="515" alt="image" src="https://github.com/user-attachments/assets/5d384af8-7a3d-4639-b7c1-47be5c58b139" />

Checking information on the trainin, validation and Test set.
``` python
print('\nLoading Train Data')
train_data = tf.keras.utils.image_dataset_from_directory("clothing-dataset-small/train", shuffle = True,)

print('\nLoading Validation Data')
val_data = tf.keras.utils.image_dataset_from_directory("clothing-dataset-small/validation", shuffle = False,)

print('\nLoading Test Data')
test_data = tf.keras.utils.image_dataset_from_directory("clothing-dataset-small/test", shuffle = False,)
```
<img width="412" height="232" alt="image" src="https://github.com/user-attachments/assets/5eb387f5-0a3e-4890-be3c-e10e20f51189" />

Data Preprocessing
``` python
# Data Processing Stage with resizing and rescaling operations
def preprocess(train_data, val_data, test_data, target_height=128, target_width=128):
    data_preprocess = keras.Sequential(
        name="data_preprocess",
        layers=[
            layers.Resizing(target_height, target_width),
            layers.Rescaling(1.0/255),]
    )
    train_ds = train_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_data.map(lambda x, y: (data_preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
```
``` python
train_ds, val_ds, test_ds = preprocess(train_data, val_data, test_data)
```
Model development
``` python
def arch_1(height=128, width=128):
    num_classes = 10
    hidden_size = 256

    model = keras.Sequential(
        name="model_cnn_1",
        layers=[
            layers.Conv2D(filters=16, kernel_size=3, padding="same", activation='relu', input_shape=(height, width, 3)),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=32, kernel_size=3, padding="same", activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64, kernel_size=3, padding="same", activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=128, kernel_size=3, padding="same", activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=256, kernel_size=3, padding="same", activation='relu'),
            # layers.MaxPooling2D(),
            # layers.Flatten(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(units=hidden_size, activation='relu'),
            layers.Dense(units=num_classes, activation='softmax')
        ])
    return model
```
Summary of the model
```python
model = arch_1()
model.summary()
```
<img width="641" height="696" alt="image" src="https://github.com/user-attachments/assets/149f6c57-aa77-4fc7-896e-4ebd2e64e15e" />

Compiling and fitting of the model
``` python
def compile_train_v1(model, train_ds, val_ds, ckpt_path="model_checkpoint.weights.h5",epochs = 10):
    epochs = 20
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model_fit = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[keras.callbacks.ModelCheckpoint(
                ckpt_path,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            )])
    return model_fit
```
``` python
model_fit = compile_train_v1(model, train_ds, val_ds)
```
<img width="1283" height="714" alt="image" src="https://github.com/user-attachments/assets/c4ca357e-c8db-4517-8658-0b25654520ef" />
<img width="1274" height="239" alt="image" src="https://github.com/user-attachments/assets/a42d7fbe-a502-4c12-9524-1e5c5fa2efed" />

Plot of Accuracy and loss of the model
``` python
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(15,5))
ax = axes.ravel()

# Accuracy graph
ax[0].plot(range(0,model_fit.params['epochs']), [acc * 100 for acc in model_fit.history['accuracy']], label='Train', color='b')
ax[0].plot(range(0,model_fit.params['epochs']), [acc * 100 for acc in model_fit.history['val_accuracy']], label='Val', color='r')
ax[0].set_title('Accuracy vs. epoch', fontsize=15)
ax[0].set_ylabel('Accuracy', fontsize=15)
ax[0].set_xlabel('epoch', fontsize=15)
ax[0].legend()

# loss graph
ax[1].plot(range(0,model_fit.params['epochs']), model_fit.history['loss'], label='Train', color='b')
ax[1].plot(range(0,model_fit.params['epochs']), model_fit.history['val_loss'], label='Val', color='r')
ax[1].set_title('Loss vs. epoch', fontsize=15)
ax[1].set_ylabel('Loss', fontsize=15)
ax[1].set_xlabel('epoch', fontsize=15)
ax[1].legend()
plt.show()
```
<img width="1235" height="478" alt="image" src="https://github.com/user-attachments/assets/d7c7934e-64f5-4c09-886f-13e2d6f77e0a" />

Test accuracy of our test data
``` python
from sklearn.metrics import accuracy_score, confusion_matrix

true_categories = tf.concat([y for x, y in test_ds], axis=0)
images = tf.concat([x for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
class_names = test_data.class_names
predicted_categories = tf.argmax(y_pred, axis=1)

test_acc = accuracy_score(true_categories, predicted_categories) * 100
print(f'\nTest Accuracy: {test_acc:.2f}%\n')
```
<img width="571" height="474" alt="image" src="https://github.com/user-attachments/assets/572f91c8-9a41-4a8b-9bec-5401f99a3700" />

We notice that mdoel is performing with an accuarcy of 65%. let's quiclly check how our model is perfroming with respect to the loss and accuracy of the model.

Plot of image, true label and class probabilities

``` python
# function to plot image given image, its true label and class probabilities (pred_array)
def plot_image(pred_array, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(pred_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(pred_array),
                                class_names[true_label]),
                                color=color)

# function to plot barplot of class probabilities (pred_array)
def plot_value_array(pred_array, true_label):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), pred_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(pred_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')
```
``` python
true_categories = tf.concat([y for x, y in test_ds], axis=0)
images = tf.concat([x for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
class_names = test_data.class_names

# Randomly sample 15 test images and plot it with their predicted labels, and the true labels.
indices = random.sample(range(len(images)), 15)
# Color correct predictions in green and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i,index in enumerate(indices):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(y_pred[index], true_categories[index], images[index])
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(y_pred[index], true_categories[index])
plt.tight_layout()
plt.show()
```
<img width="1177" height="989" alt="image" src="https://github.com/user-attachments/assets/6e3be21f-75cd-4ca6-b038-db6844ebcfce" />

**Fine tuning hyperparameters**
``` python
def arch_2(height=128, width=128):
    num_classes = 10
    hidden_size = 256

    model = keras.Sequential(
        name="model_cnn_2",
        layers=[
            layers.Conv2D(filters=16, kernel_size=3, padding="same", input_shape=(height, width, 3),
                            kernel_regularizer=regularizers.l2(1e-3)),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(filters=32, kernel_size=3, padding="same",
                            kernel_regularizer=regularizers.l2(1e-3)),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(filters=64, kernel_size=3, padding="same",
                            kernel_regularizer=regularizers.l2(1e-3)),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(filters=128, kernel_size=3, padding="same",
                            kernel_regularizer=regularizers.l2(1e-3)),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(filters=256, kernel_size=3, padding="same",
                            kernel_regularizer=regularizers.l2(1e-3)),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(units=hidden_size, kernel_regularizer=regularizers.l2(1e-3)),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(units=num_classes, activation='softmax')
        ]
    )
    return model
```
Fine tuned model summary
``` python
model = arch_2()
model.summary()
```
<img width="642" height="712" alt="image" src="https://github.com/user-attachments/assets/b502d4f4-cfa5-4f8d-988f-eb40476753af" /><img width="609" height="695" alt="image" src="https://github.com/user-attachments/assets/d59c8378-a49f-4938-8769-d68f0f895a62" /><img width="646" height="217" alt="image" src="https://github.com/user-attachments/assets/9f620674-6b45-48fd-9ac1-c003c0ada8ce" />

Model compiling and training
``` python
def compile_train_v2(model, train_ds, val_ds, epochs=10, ckpt_path="model_checkpoint.weights.h5"):
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=0.00001),

        keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True),

        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode='min')
    ]
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model_fit = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model_fit
```
``` python
model = arch_2()
model_fit = compile_train_v2(model, train_ds, val_ds, epochs=100)
```
<img width="1395" height="710" alt="image" src="https://github.com/user-attachments/assets/4ae60155-f1fd-4409-8365-423ca0eb5aea" /><img width="1446" height="687" alt="image" src="https://github.com/user-attachments/assets/5be621b9-875e-48a7-9887-e7d455b846fa" /><img width="1446" height="708" alt="image" src="https://github.com/user-attachments/assets/5201cb8b-aac9-486d-b230-f88b242edc13" /><img width="1445" height="717" alt="image" src="https://github.com/user-attachments/assets/240da6e3-5ce4-4b17-ac63-4a122e0b7217" />

Now, let's look into the plot of accuracy & plot of the mdoel we have fine tuned it.
``` python
from sklearn.metrics import classification_report
def plot_history(history):
    epochs = range(1, len(history.history['loss']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(epochs, [a * 100 for a in history.history['accuracy']], label='Train', color='b')
    if 'val_accuracy' in history.history:
        ax[0].plot(epochs, [a * 100 for a in history.history['val_accuracy']], label='Val', color='r')
    ax[0].set_title('Accuracy vs. epoch'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('Accuracy'); ax[0].legend()

    ax[1].plot(epochs, history.history['loss'], label='Train', color='b')
    if 'val_loss' in history.history:
        ax[1].plot(epochs, history.history['val_loss'], label='Val', color='r')
    ax[1].set_title('Loss vs. epoch'); ax[1].set_xlabel('epoch'); ax[1].set_ylabel('Loss'); ax[1].legend()
    plt.show()
plot_history(model_fit)
```
<img width="1238" height="470" alt="image" src="https://github.com/user-attachments/assets/0f382f16-63e4-44b6-bc24-98d8d19bcac0" />

Also, look into the accuracy of our fine tuned model
``` python
from sklearn.metrics import accuracy_score, confusion_matrix

true_categories = tf.concat([y for x, y in test_ds], axis=0)
images = tf.concat([x for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
class_names = test_data.class_names
predicted_categories = tf.argmax(y_pred, axis=1)

test_acc = accuracy_score(true_categories, predicted_categories) * 100
print(f'\nTest Accuracy: {test_acc:.2f}%\n')
```
<img width="534" height="82" alt="image" src="https://github.com/user-attachments/assets/dc6c2b9f-bb25-4e73-be3b-a6a7598d8023" />

We notice that our model is perroming better from 65% to 75%, which is a good sign of perromance on our model.
Let's look into the confusion matrix of the data.
``` python
def ConfusionMatrix(model, ds, label_list):
    y_pred = model.predict(ds)
    predicted_categories = tf.argmax(y_pred, axis=1)
    true_categories = tf.concat([y for x, y in ds], axis=0)
    cm = confusion_matrix(true_categories,predicted_categories) # last batch
    sns.heatmap(cm, annot=True, xticklabels=label_list, yticklabels=label_list, cmap="YlGnBu", fmt='g')
    plt.show()

ConfusionMatrix(model, test_ds, test_data.class_names)
```
<img width="571" height="474" alt="image" src="https://github.com/user-attachments/assets/862d022b-563e-4e50-8a61-0e7b9f4db16b" />

Let's confirm with the visualzaition of prediction of our images and with respect to the class labels as shown below.

``` python
true_categories = tf.concat([y for x, y in test_ds], axis=0)
images = tf.concat([x for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)
class_names = test_data.class_names

# Randomly sample 15 test images and plot it with their predicted labels, and the true labels.
indices = random.sample(range(len(images)), 15)
# Color correct predictions in green and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i,index in enumerate(indices):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(y_pred[index], true_categories[index], images[index])
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(y_pred[index], true_categories[index])
plt.tight_layout()
plt.show()
```
<img width="1177" height="989" alt="image" src="https://github.com/user-attachments/assets/5a19120f-b351-4b6d-8c58-dea67d143ed0" />

### Insights

- Reproducibility & Consistency: The project implements a fixed random seed (111), which ensures that the results are reproducible across different training sessions, making the experimental findings scientifically valid.

- Standardized Data Pipeline: The use of Rescaling(1./255) and Resizing layers directly within the model architecture ensures that all input data is normalized to a common scale (0 to 1), which typically helps neural networks converge faster.

- Overfitting Mitigation: The model incorporates L2 regularization (regularizers.l2(0.01)) in its dense layers. This insight suggests that the initial models may have suffered from high variance or overfitting, necessitating penalty terms to keep the weights small.

- Visual Interpretability: The inclusion of a 15-sample visualization grid with color-coded labels (green for correct, red for incorrect) provides immediate qualitative feedback on where the model struggles, such as confused classes or poor lighting conditions.

### Recommendations

- Transition to Convolutional Neural Networks (CNNs): The current architecture relies on flattening image data into a single vector for Dense layers. To significantly improve performance, it is recommended to implement Conv2D and MaxPooling2D layers, which are specifically designed to capture spatial hierarchies and local patterns in images.

- Implement Data Augmentation: To improve the model's ability to generalize to new data, you should incorporate data augmentation layers (such as RandomFlip, RandomRotation, and RandomZoom) during the preprocessing phase. This helps the model become invariant to orientation and positioning.

- Leverage Transfer Learning: Instead of training from scratch, consider using a pre-trained model like MobileNetV2 or ResNet50 as a feature extractor. This would provide a more robust starting point, especially if the dataset size is limited.

- Optimize Hyperparameters: It is recommended to implement a Learning Rate Scheduler or use EarlyStopping callbacks. This would prevent the model from overshooting the global minimum and save computational resources by stopping training once the validation loss stops improving.

- Evaluate with a Confusion Matrix: Beyond visual sampling, generating a full confusion matrix would provide a quantitative insight into which specific classes are being misclassified as others, allowing for more targeted data collection or feature engineering for those specific categories.
