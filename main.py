import argparse
from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

COMPETITION_NAME = 'competition-2026-fait-main'


def configure_gpu() -> None:
    build_info = tf.sysconfig.get_build_info()
    print('TensorFlow GPU build info:')
    print(' - CUDA version:', build_info.get('cuda_version'))
    print(' - cuDNN version:', build_info.get('cudnn_version'))
    print(' - is CUDA build:', build_info.get('is_cuda_build'))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('GPU(s) detected:')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(' -', gpu.name)
            except RuntimeError as e:
                print(' - GPU configuration failed:', e)
    else:
        print('Aucun GPU détecté. Le code fonctionnera sur CPU.')

CLASS_NAMES = [
    'coastguard_scaled',
    'containership_scaled',
    'corvette_scaled',
    'cruiser_scaled',
    'cv_scaled',
    'destroyer_scaled',
    'methanier_scaled',
    'smallfish_scaled',
    'submarine_scaled',
    'tug_scaled',
]
CLASS_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


def locate_competition_data() -> Path:
    cache_root = Path.home() / '.cache' / 'kagglehub' / 'competitions' / COMPETITION_NAME
    if cache_root.exists():
        print('Competition cache found at', cache_root)
        return cache_root

    print('Downloading competition dataset...')
    dataset_path = Path(kagglehub.competition_download(COMPETITION_NAME))
    print('Downloaded dataset to', dataset_path)
    return dataset_path


def make_datasets(base_dir: Path, image_size=(128, 192), batch_size=32, val_split=0.15, seed=42):
    train_dir = base_dir / 'ships24' / 'ships_gray' / 'ships_gray'
    if not train_dir.exists():
        raise FileNotFoundError(f'Training directory not found: {train_dir}')

    common_kwargs = dict(
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        validation_split=val_split,
        seed=seed,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        subset='training',
        **common_kwargs,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        subset='validation',
        **common_kwargs,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds


def build_model(input_shape=(128, 192, 1), num_classes=10):
    l2_reg = regularizers.l2(1e-5)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255.0),
        layers.GaussianNoise(0.05),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.12),
        layers.RandomTranslation(height_factor=0.09, width_factor=0.09),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),

        layers.Conv2D(32, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        layers.Conv2D(256, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.35),

        layers.Conv2D(512, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.Conv2D(512, 3, padding='same', activation='swish', kernel_regularizer=l2_reg),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='swish', kernel_regularizer=l2_reg),
        layers.Dropout(0.5),
        layers.Dense(256, activation='swish', kernel_regularizer=l2_reg),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


def compute_class_weight(train_dir: Path) -> dict[int, float]:
    counts = []
    for class_name in CLASS_NAMES:
        class_path = train_dir / class_name
        counts.append(len([p for p in class_path.iterdir() if p.is_file()]))

    total = sum(counts)
    num_classes = len(CLASS_NAMES)
    return {
        index: total / (num_classes * count) if count > 0 else 1.0
        for index, count in enumerate(counts)
    }


def count_non_activation_layers(model: tf.keras.Model) -> int:
    return sum(
        1 for layer in model.layers
        if not isinstance(layer, layers.Activation)
    )


def load_test_images(base_dir: Path):
    test_path = base_dir / 'ships24' / 'test_images.npy'
    if not test_path.exists():
        raise FileNotFoundError(f'Test file not found: {test_path}')
    images = np.load(test_path)
    if images.ndim != 3:
        raise ValueError(f'Expected test images shape (N, H, W), got {images.shape}')
    return images.astype(np.float32)


def save_predictions(predictions: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as csv_file:
        csv_file.write('ID,Category\n')
        for idx, label in enumerate(predictions.tolist()):
            csv_file.write(f'{idx},{int(label)}\n')
    print('Saved submission to', output_path)


def main():
    parser = argparse.ArgumentParser(description='Train a CNN for ship classification and create submission.')
    parser.add_argument('--epochs', type=int, default=70, help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--model-path', type=Path, default=Path('ship_classifier.h5'), help='Path to save the model')
    parser.add_argument('--submission-path', type=Path, default=Path('test.csv'), help='Path for submission CSV')
    args = parser.parse_args()

    configure_gpu()

    competition_dir = locate_competition_data()
    train_ds, val_ds = make_datasets(competition_dir, batch_size=args.batch_size)
    train_dir = competition_dir / 'ships24' / 'ships_gray' / 'ships_gray'
    class_weight = compute_class_weight(train_dir)

    print('Training dataset classes:', CLASS_NAMES)
    print('Dataset class mapping:', CLASS_MAP)
    print('Training batches:', len(train_ds), 'Validation batches:', len(val_ds))
    print('Class weights:', class_weight)

    model = build_model()
    print('Nombre de couches :', count_non_activation_layers(model))
    print("Nombre exact de couches dans model.layers :", len(model.layers))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3.5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(args.model_path),
        save_best_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        verbose=1,
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_sparse_categorical_accuracy',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=[checkpoint, early_stop, reduce_lr],
    )

    val_loss, val_acc = model.evaluate(val_ds)
    print(f'Validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}')

    model.save(args.model_path)

    test_images = load_test_images(competition_dir)
    test_images = np.expand_dims(test_images, axis=-1)
    test_images = test_images.astype(np.float32)

    print('Test images shape:', test_images.shape)
    predictions = model.predict(test_images, batch_size=args.batch_size)
    predicted_labels = np.argmax(predictions, axis=1)

    unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
    print('Prediction distribution:', dict(zip(unique_labels.tolist(), label_counts.tolist())))
    if unique_labels.size == 1:
        print('Attention : toutes les prédictions sont pour la même classe ->', unique_labels[0])

    save_predictions(predicted_labels, args.submission_path)
    print('Done.')


if __name__ == '__main__':
    main()