import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras import layers, models
import pickle

# --- SETTINGS ---
base_dir = r"C:\Users\zezom\PycharmProjects\HorusEye"
dl_ready_dir = os.path.join(base_dir, "Data", "Processed", "DL_ready")
labels_path = os.path.join(base_dir, "train.csv")
model_dir = os.path.join(base_dir, "Models")
os.makedirs(model_dir, exist_ok=True)

# --- Load competition labels ---
labels_df = pd.read_csv(labels_path)
print(f"[INFO] Loaded {len(labels_df)} rows from train.csv")

# --- Filter for MI only ---
# Try to detect MI samples by a 'paradigm' column. If not found, you can filter by label naming if possible.
if 'paradigm' in labels_df.columns:
    mi_df = labels_df[labels_df['paradigm'].str.lower().str.contains('mi')]
    print(f"[INFO] Filtered for MI only: {len(mi_df)} rows remain.")
else:
    # If there's no 'paradigm', you must adapt the filter to your dataset
    # For example, if MI labels are 0/1/2/3 and SSVEP are 4/5/6/7:
    # mi_df = labels_df[labels_df['label'].isin([0, 1, 2, 3])]
    print("[WARN] No 'paradigm' column found, using all rows. Update this section if you want to filter for MI only.")
    mi_df = labels_df

X = []
y = []
missing_files = []
skipped_ssvep = 0

for i, row in mi_df.iterrows():
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
        print(f"[WARN] Trial number {trial_num} out of range for {npy_path} (max trial: {epochs.shape[0]})")
        continue

    # If MI-only data is present, but some CSV rows refer to SSVEP trials, skip those
    try:
        X.append(epochs[trial_num - 1])  # Select the trial data
        y.append(label)
    except Exception as ex:
        print(f"[WARN] Could not load trial {trial_num} from {npy_path}: {ex}")
        skipped_ssvep += 1

print(f"[INFO] Finished loading. Samples loaded: {len(X)}")
if missing_files:
    print(f"[WARN] {len(missing_files)} files missing (showing 3): {missing_files[:3]}")
if skipped_ssvep:
    print(f"[WARN] Skipped {skipped_ssvep} rows (possible SSVEP or trial mismatch)")

if len(X) == 0:
    print("[ERROR] No data loaded! Check warnings above and your filters/paths.")
    exit()

X = np.stack(X)  # (num_trials, n_channels, n_samples)
print(f"[INFO] Data shape after stacking: {X.shape}")

# --- Normalize per channel (important for EEG deep learning) ---
X = (X - np.mean(X, axis=(0, 2), keepdims=True)) / (np.std(X, axis=(0, 2), keepdims=True) + 1e-8)

# --- Encode labels ---
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
print(f"[INFO] Unique classes: {label_encoder.classes_}")

# --- Transpose to (samples, timesteps, channels) ---
X = np.transpose(X, (0, 2, 1))  # (samples, timesteps, channels)
print(f"[INFO] Transposed data shape: {X.shape}")

# --- One-hot encode labels ---
num_classes = len(np.unique(y_enc))
y_cat = to_categorical(y_enc, num_classes=num_classes)

# --- Train/Val Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"[INFO] Train/val split: {X_train.shape}, {X_val.shape}")


# --- EEG-Transformer Model ---
def eeg_transformer(
        input_shape,
        num_classes,
        num_heads=8,
        ff_dim=128,
        num_transformer_blocks=2,
        dropout=0.3
):
    inputs = layers.Input(shape=input_shape)  # (timesteps, channels)

    # Channel-wise embedding (1x1 conv)
    x = layers.Conv1D(32, 1, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Positional encoding (temporal)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_encoding = layers.Embedding(input_dim=input_shape[0], output_dim=32)(positions)
    x = x + pos_encoding

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head Self-Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=32, dropout=dropout
        )(x, x)
        attn_output = layers.BatchNormalization()(attn_output)
        # Add & Norm
        x = layers.Add()([x, attn_output])
        # Feedforward
        ff = layers.Dense(ff_dim, activation='relu')(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(32)(ff)
        ff = layers.BatchNormalization()(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    # Global pooling and classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)


# --- Build Model ---
input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, channels)
model = eeg_transformer(input_shape=input_shape, num_classes=num_classes)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
)

# --- Save model and label encoder for inference ---
model.save(os.path.join(model_dir, "eeg_transformer_model.h5"))
with open(os.path.join(model_dir, "label_encoder.pkl"), 'wb') as f:
    pickle.dump(label_encoder, f)

print("[INFO] Training complete. Model and label encoder saved.")
