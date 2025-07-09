# Bildklassifikation der Simpsons-Charaktere mit Neuronalen Netzen

# Name: Justin Stange-Heiduk  
# Matrikelnummer: [Deine Matrikelnummer]  
# Universität: AKAD   
# Kurs: B.Sc Data Science     
# Dozent: Dr. Martin Prause   
# Beginn: 03.02.2025  
# Orientiert an: https://www.kaggle.com/code/serkan0yldz/cnn-classification-for-simpsons-characters-dataset

#FYI: Musste das Modell Training in einer Python Datei statt in einen Jyupter Notebook durchführen, da es sonst zu einem Fehler kam bei der Verwendung meiner GPU mit Tensorflow.

# Allgemeine Bibliotheken
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Fortschrittsbalken für Schleifen
from collections import Counter

import numpy as np
import shutil

# Bildverarbeitung
from PIL import Image  # Pillow für Bildmanipulation

# Machine Learning / Deep Learning
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, ReLU, LeakyReLU, ELU 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import keras_tuner as kt
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight


# Vortrainiertes VGG16-Modell laden (ohne Top-Schichten)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


## Automatische Architektur mit Keras Tuner

# InceptionV3 als Feature-Extractor verwenden.
# Extrahierte Features in Dense-Schicht überführen
# Verschiedene Aktivierungsfunktionen ausprobieren ("relu", "leaky_relu", "elu")
# Dynamische Anzahl von Neuronen in der Dense-Schicht testen (z. B. 32–1024).
# Testen ob noch eine Dense Schicht sinn macht.
# Verschiedene Lernraten ausprobieren (1e-3 bis 1e-5).
# Dropout & Batch Normalization nutzen, um Overfitting zu verhindern.

#########################################################################################################################################

# Dynamische Modellfunktion für Keras Tuner
def build_model(hp):

    # Wähle, ob Fine-Tuning erlaubt ist (z. B. die letzten `N` Schichten trainierbar machen)
    fine_tune_at = hp.Int("fine_tune_layers", min_value=1, max_value=3, step=1)


    # Feature-Extraktion bleibt eingefroren, aber Fine-Tuning für letzte `N` Schichten aktivieren
    for layer in base_model.layers[:-fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_at:]:
        layer.trainable = True

    # Feature-Extraktion durch VGG16
    x = base_model.output

    # Dynamische `Conv2D`-Schicht nach VGG16 (statt feste `64` Filter)
    filters = hp.Choice("filters", values=[32, 64, 128])
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation="relu", padding="same", name="custom_conv2d")(x)
    x = MaxPool2D(pool_size=(2, 2), name="custom_maxpool")(x)

    # Flatten für die Dense-Schicht
    x = Flatten(name="flatten")(x)

    # Haupt-Dense-Schicht mit `L2-Regularisierung`
    hp_units = hp.Choice("units", values=[256, 512, 1024])
    x = Dense(hp_units, kernel_regularizer=l2(1e-4), name="dense_layer")(x)

    # Aktivierungsfunktion setzen
    x = ReLU(name="activation_relu")(x)

    # BatchNormalization + Dropout für bessere Regularisierung
    x = BatchNormalization(name="batch_norm")(x)
    dropout_rate = hp.Choice("dropout_rate", values=[0.3, 0.5])
    x = Dropout(rate=dropout_rate, name="dropout")(x)
    # Softmax-Output für 13 Klassen
    x = Dense(13, activation="softmax", name="output")(x)

    # Lernrate für den Optimizer
    learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])

    # Modell kompilieren mit `accuracy`
    model = Model(inputs=base_model.input, outputs=x, name="VGG16_Custom")

    # Modell kompilieren
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        metrics=["accuracy"]
)

    return model


# Keras Tuner mit Hyperband-Optimierung
tuner = kt.Hyperband(
    build_model,  # Modellfunktion
    objective='val_accuracy',  # Maximiert die Validierungsgenauigkeit
    max_epochs=5,  # Maximale Anzahl der Epochen pro Versuch
    factor=3,  # Schlechte Modelle werden früh gestoppt
    hyperband_iterations=2  # Anzahl der Hyperband-Zyklen
)


# Callbacks für besseres Training
# EarlyStopping verhindert Overfitting, indem es das Training automatisch stoppt.
# ReduceLROnPlateau sorgt dafür, dass das Modell sich weiter verbessert, wenn val_loss stagniert
tuner_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),  # Stoppt, wenn `val_loss` sich 5 Epochen lang nicht verbessert
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)  # Reduziert Lernrate bei Stagnation
]

#########################################################################################################################################

# Ordner definieren
input_dir = "./filtered_simpsons_dataset"
output_dir = "./simpsons_dataset_split"

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

IMAGE_SIZE=224
BATCH_SIZE=32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    seed=42,
    label_mode="int",
    labels="inferred",
    color_mode="rgb",
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    seed=42,
    label_mode="int",
    labels="inferred",
    color_mode="rgb",
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    seed=42,
    label_mode="int",
    labels="inferred",
    color_mode="rgb",
)

class_names = train_dataset.class_names

# Klassen-Labels abrufen
train_labels_np = np.concatenate([y.numpy() for _, y in train_dataset])  # Labels aus Dataset extrahieren

# Klassengewichte berechnen (basierend auf tatsächlichen Trainingslabels!)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels_np),  # Integer-Labels verwenden
    y=train_labels_np
)

# Dictionary für Keras Training
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Ausgabe der berechneten Klassengewichte mit Klassennamen
print("Berechnete Klassengewichte:")
for class_index, weight in class_weight_dict.items():
    print(f"Klasse {class_index} ({class_names[class_index]}): {weight:.4f}")


train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (preprocess_input(x), y))
test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y))


# Starte die Hyperparameter-Suche mit GPU-Unterstützung
########################################################################################################################################
tuner.search(
    train_dataset,  # Trainingsdaten
    epochs=5,  # Maximale Epochen für jedes getestete Modell
    validation_data=validation_dataset,  # Validierungsdaten für Hyperparameter-Optimierung
    callbacks=tuner_callbacks,  # Callbacks für stabileres Training
    class_weight=class_weight_dict  # Falls Klassengewichte berechnet wurden
)
#Speicher den Tuner nach der Hyperparameter-Suche
tuner.save()
tuner.results_summary()
print("✅ Hyperparameter-Tuner gespeichert!")

# Best Hyperparameters aus Tuner extrahieren
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

fine_tune_layers = best_hps.get("fine_tune_layers")
filters = best_hps.get("filters")
units = best_hps.get("units")
dropout_rate = best_hps.get("dropout_rate")
learning_rate = best_hps.get("learning_rate")

print("Beste Hyperparameter hier in meinen kopf:")  # Ausgabe der besten Hyperparameter
print(f"fine_tune_layers: {fine_tune_layers}")
print(f"filters: {filters}")
print(f"units: {units}")
print(f"dropout_rate: {dropout_rate}")
print(f"learning_rate: {learning_rate}")
########################################################################################################################################


# Vortrainiertes VGG16-Modell laden (ohne Top-Schichten)
base_model_ = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Fine-Tuning aktivieren
for layer in base_model_.layers[:-fine_tune_layers]:
    layer.trainable = False
for layer in base_model_.layers[-fine_tune_layers:]:
    layer.trainable = True

# Feature-Extraktion durch VGG16
X = base_model_.output

# Dynamische `Conv2D`-Schicht nach VGG16 mit optimierten Parametern
X = Conv2D(filters=filters, kernel_size=(3, 3), activation="relu", padding="same", name="custom_conv2d")(X)
X = MaxPool2D(pool_size=(2, 2), name="custom_maxpool")(X)

# Flatten für die Dense-Schicht
X = Flatten(name="flatten")(X)

# Haupt-Dense-Schicht mit `L2-Regularisierung`
X = Dense(units, kernel_regularizer=l2(1e-4), name="dense_layer")(X)

# Aktivierungsfunktion ReLU
X = ReLU(name="activation_relu")(X)

# BatchNormalization + Dropout für bessere Regularisierung
X = BatchNormalization(name="batch_norm")(X)
X = Dropout(rate=dropout_rate, name="dropout")(X)

# Softmax-Output für 13 Klassen
X = Dense(13, activation="softmax", name="output")(X)

# Modell erstellen
model = Model(inputs=base_model_.input, outputs=X, name="VGG16_Custom_")

# Modell kompilieren
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

# Zusammenfassung des Modells anzeigen
model.summary()

# Trainings-Callbacks für das Modell definieren
# Frühes Stoppen, falls sich `val_loss` 5 Epochen lang nicht verbessert
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,  # Stoppt, wenn sich `val_loss` für 5 Epochen nicht verbessert
    restore_best_weights=True  # Nimmt das beste Modell, nicht das letzte
)

# Reduzierung der Lernrate, falls `val_loss` stagniert
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,  # Reduziert die Lernrate auf 50% des aktuellen Werts
    patience=2,  # Falls sich `val_loss` für 2 Epochen nicht verbessert
    min_lr=1e-6  # Mindest-Lernrate, um nicht zu niedrig zu gehen
)

# Speichert das beste Modell während des Trainings
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model_training.h5",  # Dateiname
    monitor="val_loss",
    save_best_only=True,  # Speichert nur das beste Modell
    verbose=1  # Zeigt an, wenn ein neues bestes Modell gespeichert wurde
)

# TensorBoard für Visualisierungen
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir="logs",  # Speicherort der Logs
    histogram_freq=1,  # Aktiviert Histogramme für Gewichtsverteilungen
    write_graph=True,  # Speichert das Modell für Visualisierungen
    update_freq="epoch"  # Aktualisiert die Logs nach jeder Epoche
)


#  Liste aller Callbacks
training_callbacks = [early_stopping, reduce_lr, model_checkpoint, tensorboard]

# Performance optimieren
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Training starten
history = model.fit(
    train_dataset,
    epochs=50,  # Maximale Anzahl der Epochen
    validation_data=validation_dataset,
    callbacks=training_callbacks,  # Callbacks aktivieren
    class_weight=class_weight_dict  # Falls Klassengewichte berechnet wurden
)


# Modell speichern
model.save("best_model_trainiert_simpsons.h5")
print("✅ Modell erfolgreich gespeichert als 'best_model_trainiert_simpsons.h5'!")
