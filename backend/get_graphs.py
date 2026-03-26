import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print(" 1. Loading your saved model...")
model = tf.keras.models.load_model('xray_model.h5')

print(" 2. Loading your dataset images...")
# Make sure 'dataset_flat' or 'dataset' matches the folder name where your images are!
dataset_path = 'dataset_flat' if os.path.exists('dataset_flat') else 'dataset'

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(160, 160),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

print(" 3. Testing the model (this takes about 60 seconds)...")
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_names = ['COVID-19', 'Normal', 'Pneumonia']

print(" 4. Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show() # This pops up the first graph!

print(" 5. Generating ROC Curve...")
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr, tpr, _ = roc_curve(y_true_bin[:, 0], predictions[:, 0])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='red', lw=2, label=f'COVID-19 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show() # This pops up the second graph!