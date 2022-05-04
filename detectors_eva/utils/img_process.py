import cv2
import numpy as np




def preprocess_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed



def to_gray_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)

    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    normalized_images = images.mean(1).unsqueeze(1)
    return normalized_images


def to_color_normalized(images):
    """Performs image normalization and converts images to grayscale (preserving dimensions)

    Parameters
    ----------
    images: torch.Tensor
        Input images.

    Returns
    -------
    normalized_images: torch.Tensor
        Normalized grayscale images.
    """
    assert len(images.shape) == 4
    images -= 0.5
    images *= 0.225
    return images

