import tensorflow as tf
from tensorflow.keras import layers, models
import random
import os
# Oxford IIIT Cats: https://www.kaggle.com/datasets/imbikramsaha/cat-breeds

images_path = "CatsDataset"
batch_size = 32
img_size = 400  # Images will be converted to 400 x 400 before input into the model
seed = random.randrange(1, 100)  # Or set manually
test_split = 0.2

print("Seed:", seed)

# Load dataset and split into test and train sets
train_set = tf.keras.utils.image_dataset_from_directory(
    images_path,
    validation_split=test_split,
    subset="training",
    seed=seed,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

test_set = tf.keras.utils.image_dataset_from_directory(
    images_path,
    validation_split=test_split,
    subset="validation",
    seed=seed,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

# Cache dataset to disk
train_set = train_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_set = test_set.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

num_classes = 12

# CNN Model
model = models.Sequential()
model.add(layers.RandomFlip("horizontal_and_vertical"))
model.add(layers.RandomRotation(0.2))
model.add(layers.RandomZoom((0, 0.2)))
model.add(layers.Rescaling(1. / 255))

model.add(layers.Conv2D(8, 3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(16, 3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64, 3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(92, 3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

num_epochs = 50

# Define custom callback to save checkpoint every 5 epochs
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_path, save_every_n_epoch):
        super().__init__()
        self.save_path = save_path
        self.save_every_n_epoch = save_every_n_epoch
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every_n_epoch == 0:
            filepath = self.save_path.format(epoch=epoch + 1)
            self.model.save_weights(filepath)
            print(f"Checkpoint saved at {filepath}")

cp_callback = CustomModelCheckpoint(
    save_path="Checkpoint/cp_epoch_{epoch}.ckpt",
    save_every_n_epoch=5
)

# Train model
model.fit(
    train_set,
    validation_data=test_set,
    epochs=num_epochs,
    callbacks=[cp_callback]
)

# Save model to disk
model.save("CatClassifier.keras")
print("Model saved.")

# To load later:
# model = tf.keras.models.load_model("CatClassifier.keras")
