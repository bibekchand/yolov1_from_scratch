import os
import numpy as np
from PIL import Image
x, y = [], []

part = 0


def save_and_clear_arrays():
    global part, x, y
    x_numpy = np.array(x, dtype=np.float32).transpose(0, 3, 1, 2)
    y_numpy = np.array(y, dtype=np.float32).reshape(-1, 1)
    np.savez(f'cat_dogs_part{part}.npz', x=x_numpy, y=y_numpy)
    print('saved cat_dogs_part', part, '.npz', 'with', len(x), len(y),)
    del x_numpy, y_numpy
    y.clear()
    x.clear()
    part += 1


def iterate_through(folder_path, cat):
    files = sorted(os.listdir(folder_path))
    i = 0
    for f in files:
        if (i == 1000):
            save_and_clear_arrays()
            i = 0
        if f.endswith('db'):
            continue
        full_file_path = os.path.join(folder_path, f)
        try:
            img = Image.open(full_file_path)
        except Exception as e:
            print("skipped because of", e, "file", f)
            continue
        img = img.resize((300, 300))
        img_array = np.array(img, dtype=np.float32)
        if img_array.shape != (300, 300, 3):
            continue
        x.append(img_array)
        if cat:
            y.append(0)
        else:
            y.append(1)
        i += 1
    save_and_clear_arrays()
