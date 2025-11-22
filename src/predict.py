# src/predict.py
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
import os

# Optional: set this to True to save intermediate images for debugging
DEBUG_SAVE = False
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

def _save_debug(img: Image.Image, name: str):
    if DEBUG_SAVE:
        path = os.path.join(DEBUG_DIR, name)
        img.save(path)

def _center_image_by_com(np_img):
    arr = np_img.astype(np.float32)
    total = arr.sum()
    if total == 0:
        return np_img

    coords_y = np.arange(arr.shape[0])
    coords_x = np.arange(arr.shape[1])
    com_y = (arr.sum(axis=1) * coords_y).sum() / total
    com_x = (arr.sum(axis=0) * coords_x).sum() / total

    c_y, c_x = (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0

    shift_y = int(round(c_y - com_y))
    shift_x = int(round(c_x - com_x))

    shifted = np.zeros_like(arr)

    src_y0 = max(0, -shift_y)
    src_y1 = min(arr.shape[0], arr.shape[0] - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(arr.shape[1], arr.shape[1] - shift_x)

    dst_y0 = max(0, shift_y)
    dst_y1 = min(arr.shape[0], arr.shape[0] + shift_y)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(arr.shape[1], arr.shape[1] + shift_x)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return shifted

def preprocess_image(img: Image.Image):
    img = img.convert("L")
    arr0 = np.array(img)
    mean_val = arr0.mean()

    if mean_val > 127:
        img = ImageOps.invert(img)
        arr0 = np.array(img)

    _save_debug(Image.fromarray(arr0), "step0_inversion_check.png")

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    _save_debug(img, "step1_cropped.png")

    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    _save_debug(img, "step2_thumbnail.png")

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - img.size[0]) // 2
    top = (28 - img.size[1]) // 2
    canvas.paste(img, (left, top))
    _save_debug(canvas, "step3_pasted_28x28_before_com.png")

    arr = np.array(canvas).astype(np.float32)

    pil_canvas = Image.fromarray(arr.astype(np.uint8))
    pil_canvas = pil_canvas.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(pil_canvas).astype(np.float32)

    arr_centered = _center_image_by_com(arr)
    _save_debug(Image.fromarray(arr_centered.astype(np.uint8)), "step4_centered_by_com.png")

    arr_final = arr_centered / 255.0
    arr_final = np.expand_dims(arr_final, axis=(0, -1)).astype("float32")
    return arr_final

def predict_image(model, pil_img):
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs

if __name__ == "__main__":
    # Correct relative path when running predict.py directly
    model_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "mnist_cnn.h5")
    model_path = os.path.abspath(model_path)

    print("Loading model from:", model_path)
    model = tf.keras.models.load_model(model_path)

    test_path = "some_digit.png"
    if os.path.exists(test_path):
        img = Image.open(test_path)
        pred, probs = predict_image(model, img)
        print("Predicted:", pred)
        print("Probs:", probs)
    else:
        print("No test image found at some_digit.png")
