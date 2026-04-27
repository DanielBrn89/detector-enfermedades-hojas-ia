from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/processed/tomato")
EXTENSIONS = [".jpg", ".jpeg", ".png"]

bad_files = []

for img_path in DATA_DIR.rglob("*"):
    if img_path.suffix.lower() in EXTENSIONS:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            bad_files.append((img_path, e))

print(f"Imágenes dañadas encontradas: {len(bad_files)}")

for path, error in bad_files:
    print(path)
    print(error)