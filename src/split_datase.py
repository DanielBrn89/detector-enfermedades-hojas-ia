import os
import shutil
import random

# Rutas
SOURCE_DIR = "data/processed/tomato"
DEST_DIR = "data/processed_split"

# Porcentajes
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


def split_data():
    classes = os.listdir(SOURCE_DIR)

    for cls in classes:
        class_path = os.path.join(SOURCE_DIR, cls)

        if not os.path.isdir(class_path):
            continue

        images = [
            file for file in os.listdir(class_path)
             if os.path.isfile(os.path.join(class_path, file))
             and file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * TRAIN_SPLIT)
        val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        for split, files in zip(
            ["train", "val", "test"],
            [train_files, val_files, test_files]
        ):
            split_path = os.path.join(DEST_DIR, split, cls)
            os.makedirs(split_path, exist_ok=True)

            for file in files:
                src = os.path.join(class_path, file)
                dst = os.path.join(split_path, file)

                shutil.copy(src, dst)

        print(f"Clase {cls} dividida correctamente.")


if __name__ == "__main__":
    split_data()