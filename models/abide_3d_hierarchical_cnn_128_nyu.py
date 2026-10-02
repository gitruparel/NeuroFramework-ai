"""
==============================================================================
ABIDE-I 3D Hierarchical Multi-Planar Conv3D Model (128x128 NYU Pipeline)
------------------------------------------------------------------------------
Author: Clint Loyed
Target Site: NYU Langone Medical Center Alone (N=184 Subjects)
Target Resolution: (50, 128, 128, 1) 3D Volumetric Tensors
Data Directory: './processed_paper_3D/'
Based on the sMRI 3D Architecture by Hammash & Younis (2026)
==============================================================================
"""

import os
import sys

# Bypass broken matplotlib import hooks in environment
class DummyMatplotlib:
    pass
sys.modules['matplotlib'] = DummyMatplotlib()
sys.modules['matplotlib.pyplot'] = DummyMatplotlib()

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.ops as K
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate, Add, 
                                     Conv3D, MaxPooling3D, GlobalAveragePooling3D, 
                                     GlobalMaxPooling3D, Activation, LayerNormalization, BatchNormalization, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("1. Initializing 128x128 3D Tensor Ecosystem for NYU Cohort...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🚀 GPU DETECTED & REGISTERED: {gpus}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
else:
    print("ℹ️ Running model training on active accelerator device...")

GLOBAL_BATCH_SIZE = 8 
EPOCHS = 50

# Output Directories & 128x128 Data Path
if os.path.exists('/kaggle/working'):
    base_dir = '/kaggle/working/'
else:
    base_dir = './'

if os.path.exists('./data/ABIDE_Phenotypic.csv'):
    phenotype_csv = './data/ABIDE_Phenotypic.csv'
elif os.path.exists('/kaggle/working'):
    phenotype_csv = '/kaggle/input/datasets/clintloyed/abide-autism-10x-data/ABIDE_Phenotypic.csv'
else:
    phenotype_csv = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv'

data_dir = os.path.join(base_dir, 'processed_paper_3D')

print(f"📁 Ingesting 128x128 NYU 3D Tensors from: '{data_dir}'...")
df = pd.read_csv(phenotype_csv)
df = df[df['SITE_ID'] == 'NYU'].dropna(subset=['DX_GROUP'])
label_dict = {str(row['SUB_ID']).zfill(7): 1 if row['DX_GROUP'] == 1 else 0 for _, row in df.iterrows()}

X_ax, X_cor, X_sag, y = [], [], [], []

for patient_id in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, patient_id)
    if not os.path.isdir(folder_path) or patient_id not in label_dict:
        continue
        
    try:
        ax = np.load(os.path.join(folder_path, "axial_50.npy"))
        cor = np.load(os.path.join(folder_path, "coronal_50.npy"))
        sag = np.load(os.path.join(folder_path, "sagittal_50.npy"))
    except Exception:
        continue
        
    X_ax.append(ax)
    X_cor.append(cor)
    X_sag.append(sag)
    y.append(label_dict[patient_id])

X_ax, X_cor, X_sag = np.array(X_ax), np.array(X_cor), np.array(X_sag)
y = np.array(y)

print(f"✅ Total 128x128 NYU 3D Volume Tensors Loaded: {len(X_ax)}")
print(f"   Shape per tensor: {X_ax.shape[1:]}")
print(f"   Autism (1): {np.sum(y == 1)}, Healthy Control (0): {np.sum(y == 0)}")

tensor_shape = X_ax.shape[1:]

def binary_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return focal_loss_fn

def build_paper_3d_cnn(input_shape):
    ax_input = Input(shape=input_shape, name='axial')
    cor_input = Input(shape=input_shape, name='coronal')
    sag_input = Input(shape=input_shape, name='sagittal')
    
    def hierarchical_block(x, filters, kernel_size):
        res = Conv3D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
        x = Conv3D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, res])
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        return x

    def build_3d_feature_extractor(view_input):
        x = hierarchical_block(view_input, 32, (3, 3, 3))
        x = hierarchical_block(x, 64, (5, 5, 5))
        x = hierarchical_block(x, 128, (3, 3, 3))
        x = hierarchical_block(x, 256, (5, 5, 5))
        
        # 3D CBAM Attention
        channel_avg = GlobalAveragePooling3D()(x)
        channel_max = GlobalMaxPooling3D()(x)
        dense_1 = Dense(256 // 8, activation='relu')
        dense_2 = Dense(256)
        
        avg_out = dense_2(dense_1(channel_avg))
        max_out = dense_2(dense_1(channel_max))
        
        channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
        channel_attention = Reshape((1, 1, 1, 256))(channel_attention)
        x = x * channel_attention
        
        spatial_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_max = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_concat = Concatenate(axis=-1)([spatial_avg, spatial_max])
        
        spatial_attention = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(spatial_concat)
        x = x * spatial_attention
        
        return GlobalAveragePooling3D()(x)

    ax_feat = build_3d_feature_extractor(ax_input)
    cor_feat = build_3d_feature_extractor(cor_input)
    sag_feat = build_3d_feature_extractor(sag_input)
    
    z = Concatenate()([ax_feat, cor_feat, sag_feat])
    z = LayerNormalization()(z)
    z = Dense(256, activation='gelu', kernel_regularizer=l2(1e-4))(z)
    z = Dropout(0.5)(z)
    predictions = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[ax_input, cor_input, sag_input], outputs=predictions)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=EPOCHS * (len(X_ax) // GLOBAL_BATCH_SIZE),
        alpha=0.01
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
                  loss=binary_focal_loss(), 
                  metrics=['accuracy'])
    return model

print("\n🚀 4. Initiating 128x128 NYU Stratified 5-Fold Cross Validation Protocol...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

def augment_3d_tensors(inputs, label):
    scale = tf.random.uniform([], 0.9, 1.1)
    flip = tf.random.uniform([], 0.0, 1.0) > 0.5
    
    def apply_aug(tensor):
        t = tensor * scale
        if flip:
            t = tf.image.flip_left_right(t)
        return t

    aug_inputs = {
        'axial': apply_aug(inputs['axial']),
        'coronal': apply_aug(inputs['coronal']),
        'sagittal': apply_aug(inputs['sagittal'])
    }
    return aug_inputs, label

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ax, y), 1):
    print(f"\n==========================================")
    print(f"🔥 TRAINING FOLD {fold} / 5 (128x128 NYU Mode)")
    print(f"==========================================")
    
    tf.keras.backend.clear_session()
    
    X_ax_t, X_ax_v = X_ax[train_idx], X_ax[val_idx]
    X_cor_t, X_cor_v = X_cor[train_idx], X_cor[val_idx]
    X_sag_t, X_sag_v = X_sag[train_idx], X_sag[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        ({'axial': X_ax_t, 'coronal': X_cor_t, 'sagittal': X_sag_t}, y_train)
    ).shuffle(len(y_train)).map(augment_3d_tensors).batch(GLOBAL_BATCH_SIZE)
    
    val_ds = tf.data.Dataset.from_tensor_slices(
        ({'axial': X_ax_v, 'coronal': X_cor_v, 'sagittal': X_sag_v}, y_val)
    ).batch(GLOBAL_BATCH_SIZE)
    
    model = build_paper_3d_cnn(tensor_shape)
    
    checkpoint_filepath = os.path.join(base_dir, f'Paper_3D_128_Fold{fold}.keras')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS, 
        callbacks=[checkpoint_callback]
    )
    
    model.load_weights(checkpoint_filepath)
    y_pred_prob = model.predict(val_ds)
    y_pred = (y_pred_prob > 0.5).astype(int)
    fold_acc = accuracy_score(y_val, y_pred)
    fold_scores.append(fold_acc)
    print(f"\n✅ Fold {fold} 128x128 Validation Accuracy: {fold_acc:.4f}")

print("\n==============================================")
print("🏆 128x128 NYU 3D MULTI-PLANAR 5-FOLD CV COMPLETE")
print("==============================================")
for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i}: {score*100:.2f}%")
print(f"🌟 FINAL 128x128 3D AVERAGE ACCURACY: {np.mean(fold_scores)*100:.2f}%")
