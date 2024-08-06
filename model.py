import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers, models
from keras.metrics import AUC

# Load data
train_labels = pd.read_csv('C:/Users/404er/Documents/kaggle/histopathologic-cancer-detection/train_labels.csv')
sample_submission = pd.read_csv('C:/Users/404er/Documents/kaggle/histopathologic-cancer-detection/sample_submission.csv')
print("Is built with CUDA:", tf.test.is_built_with_cuda())

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, using CPU")

# Inspect data
print(train_labels.info())
print(train_labels.describe())
print(train_labels.head())

# Check for missing values
missing_values = train_labels.isnull().sum()
print("Missing values in train_labels:\n", missing_values)

# Sample Submission info
print(sample_submission.info())

# Directories
train_dir = 'C:/Users/404er/Documents/kaggle/histopathologic-cancer-detection/train'
test_dir = 'C:/Users/404er/Documents/kaggle/histopathologic-cancer-detection/test'

# Visualize some of the images
def show_images(ids, labels, path, title):
    plt.figure(figsize=(15, 5))
    for i, (img_id, label) in enumerate(zip(ids, labels)):
        img_path = os.path.join(path, img_id + '.tif')
        img = Image.open(img_path)
        plt.subplot(1, len(ids), i+1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Showing 5 examples of images without cancer
show_images(train_labels[train_labels['label'] == 0]['id'][:5], [0]*5, train_dir, "Examples Without Cancer")

# Showing 5 examples of images with cancer
show_images(train_labels[train_labels['label'] == 1]['id'][:5], [1]*5, train_dir, "Examples With Cancer")

# Check number of images
num_train_images = len(os.listdir(train_dir))
num_test_images = len(os.listdir(test_dir))
total_len_of_dataset = num_train_images + num_test_images
print(f"Number of training images: {num_train_images}")
print(f"Number of test images: {num_test_images}")

# Check sample image
sample_image_path = os.path.join(train_dir, os.listdir(train_dir)[0])
sample_image = Image.open(sample_image_path)
print(f"Sample image dimensions: {sample_image.size}")
print(f"Number of channels in the sample image: {sample_image.mode}\n\n")

# Print the number of positive and negative samples in the training set
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=train_labels)
plt.title('Distribution of Labels in Training Set')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Analyze the dimensions of a sample of images to check for consistency
image_shapes = []

for img_id in train_labels['id'][:100]:  # Sample 100 images for speed
    img_path = os.path.join(train_dir, img_id + '.tif')
    img = Image.open(img_path)
    image_shapes.append(img.size)

image_shapes_df = pd.DataFrame(image_shapes, columns=['width', 'height'])

# Split the data into training and validation sets
train_data, val_data = train_test_split(train_labels, test_size=0.2, random_state=42, stratify=train_labels['label'])

# Define a fraction of the dataset to use for testing
fraction = 0.1

# Sample the training and validation IDs
train_sample_ids = np.random.choice(train_data['id'], size=int(len(train_data) * fraction), replace=False)
val_sample_ids = np.random.choice(val_data['id'], size=int(len(val_data) * fraction), replace=False)

# Create sampled dataframes
train_data_sampled = train_data[train_data['id'].isin(train_sample_ids)].copy()
val_data_sampled = val_data[val_data['id'].isin(val_sample_ids)].copy()

# Convert labels to strings for compatibility
train_data_sampled['id'] += '.tif'
val_data_sampled['id'] += '.tif'

# Convert the label column to string type
train_data_sampled['label'] = train_data_sampled['label'].astype(str)
val_data_sampled['label'] = val_data_sampled['label'].astype(str)

# Create ImageDataGenerator objects
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data_sampled,
    directory=train_dir,
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data_sampled,
    directory=train_dir,
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary'
)

# Convert the test IDs to include '.tif'
test_data = sample_submission.astype(str)
test_data['id'] += '.tif'

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=test_dir,
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,  # No labels for the test set
    shuffle=False
)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy', AUC(name='auc')])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=6
)

predictions = model.predict(
    test_generator, 
    steps=int(np.ceil(test_generator.samples / test_generator.batch_size)),  # Convert to int
    verbose=1
)

# Ensure predictions are binary labels for submission
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': sample_submission['id'],
    'label': predicted_labels
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

print("Submission CSV created: 'submission.csv'")