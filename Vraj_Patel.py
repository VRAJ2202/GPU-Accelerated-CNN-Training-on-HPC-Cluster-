import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_XLA"] = "0"



# 1. Imports & Config
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



print("=== TF Devices ===")
print(tf.config.list_physical_devices("GPU"))
print("===================")





# 2. Paths & Hyperparameters

# Folder containing 'Negative' and 'Positive' subfolders
data_dir = 'data/Concrete Crack Images for Classification'

# Hyperparameters
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42


# 3. Data Generators
#Using ImageDataGenerator simplifies loading and automatically labels images based on folder names:

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=SEED,
)



# 4. Define CNN Model (Reusable Function)
#Define your model in a function so you can easily tweak architecture:

def build_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 5. Train the Model
# Use early stopping to avoid overfitting:

model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    # callbacks=[early_stop],
    workers=1,
    use_multiprocessing=False
)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("training_history.csv", index=False)



# 6. Plot Loss & Accuracy Curves
# Single plot showing both curves:

def plot_training_curves(history):
    plt.figure(figsize=(10,5))
    
    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.savefig('Accuracy Curves.png', dpi=300, bbox_inches='tight')
    
    #plt.show()

plot_training_curves(history)


# 7. Evaluate & Confusion Matrix
# Evaluate on validation data:

# Get predictions
val_generator.reset()
preds = model.predict(val_generator, verbose=1)
pred_labels = (preds > 0.5).astype(int).flatten()
true_labels = val_generator.classes

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)

# Classification report
report = classification_report(true_labels, pred_labels, target_names=['Negative', 'Positive'])
print(report)

# Save to text file
with open("cnn_results.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)



# heatmap:


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
#plt.show()
