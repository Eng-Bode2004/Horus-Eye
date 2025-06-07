import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, LSTM, Dense, BatchNormalization
from keras.utils import to_categorical
import tensorflow as tf

# --- SETTINGS ---
base_dir = r"C:\Users\zezom\PycharmProjects\HorusEye"
dl_ready_dir = os.path.join(base_dir, "Data", "Processed", "DL_ready")
labels_path = os.path.join(base_dir, "train.csv")

# --- Load and filter labels for MI only ---
labels_df = pd.read_csv(labels_path)
mi_labels = ['Left', 'Right']  # change as needed
mi_labels_df = labels_df[labels_df['label'].isin(mi_labels)]

X = []
y = []
missing_files = []

for i, row in mi_labels_df.iterrows():
    subject = row['subject_id']
    session = row['trial_session']
    trial_num = int(row['trial'])  # 1-based
    label = row['label']

    npy_path = os.path.join(dl_ready_dir, f"{subject}_{session}_EEGdata_preprocessed_DLready.npy")
    if not os.path.exists(npy_path):
        missing_files.append(npy_path)
        continue

    epochs = np.load(npy_path)  # shape: (n_trials, n_channels, n_samples)

    if trial_num - 1 >= epochs.shape[0]:
        print(f"Warning: trial number {trial_num} out of range in file {npy_path}")
        continue

    X.append(epochs[trial_num - 1])  # Select the trial data
    y.append(label)

X = np.stack(X)  # shape: (num_trials, n_channels, n_samples)

# Encode labels
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# Transpose X to (samples, timesteps, channels) for Conv1D + LSTM input
X = np.transpose(X, (0, 2, 1))

# One-hot encode labels
num_classes = len(np.unique(y_enc))
y_cat = to_categorical(y_enc, num_classes=num_classes)

# Split into train and validation sets with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)

# Build the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))

model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]
)

if missing_files:
    print(f"Warning: {len(missing_files)} files were missing, e.g. {missing_files[:3]}")

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}")

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f"Training Accuracy: {train_acc:.4f}")
