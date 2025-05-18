import cv2

def load_and_preprocess(image_path, color_mode='grayscale'):
    if color_mode == 'grayscale':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return image

def save_image(image, path):
    if image.dtype != 'uint8':
        image = image.clip(0, 255).astype('uint8')
    cv2.imwrite(path, image)
