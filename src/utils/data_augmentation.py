"""Image augmentation using Albumentations
NOTE: comment out augmentation if you don't want to use it
"""
import albumentations as A


def aug_flip(p: float = 0.5):
    """Randomly rotate an image
    Args:
        p: probability of applying selected transform
    """
    return A.OneOf(
        [
            A.Flip(),  # randomly flip an image either horizontally, vertically or both
            A.Transpose(),  # transpose an image axis
            A.RandomRotate90(),  # randomly rotate an image by 90 degrees
            A.HorizontalFlip(),  # Flip the input horizontally around the y-axis.
            A.VerticalFlip(),  # Flip the input vertically around the x-axis.
            # A.ShiftScaleRotate()  # randomly apply affine transformation (shear and scale)
        ],
        p=p,
    )


def aug_blur(p: float = 0.5):
    """Randomly apply blur to an image
    Args:
        p: probability of applying selected transform
    """
    return A.OneOf(
        [
            A.MotionBlur(blur_limit=3),  # apply motion blur
            A.MedianBlur(blur_limit=3),  # apply median blur
            A.Blur(blur_limit=3),  # Blur the input image using a random-sized kernel.
        ],
        p=p,
    )


def aug_brightness(p: float = 0.5):
    """Randomly adjust brightness, contrast and hue/saturation/value of an image
    Args:
        p: probability of applying selected transform
    """
    return A.OneOf(
        [
            A.CLAHE(),  # apply clahe to an image
            # A.HueSaturationValue(),  #Randomly change hue, saturation and value of the input image
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=False
            ),  # randomly adjust brightness and contrast
        ],
        p=p,
    )


def aug_noise(p: float = 0.5):
    """Randomly introduce noise to an image
    Args:
        p: probability of applying selected transform
    """
    return A.OneOf(
        [
            A.GaussNoise(),  # Add gaussian noise to the input image.
        ],
        p=p,
    )


def aug_distortion(p: float = 0.5):
    """Randomly apply distortion to an image
    Args:
        p: probability of applying selected transform
    """
    return A.OneOf(
        [
            A.OpticalDistortion(),  # apply optical distrotion to an image
        ],
        p=p,
    )


def aug_nature(p: float = 0.5):
    """Randomly apply different transformation that mimic the nature
    Args:
        p: probability of applying selected transform
    """
    return A.OneOf(
        [
            A.RandomRain(),  # Adds rains effect
            A.RandomFog(),  # simulates fog for the image
            A.RandomShadow(),  # simulate shadow for the image
            A.RandomSunFlare(),  # simulates sun flares for the image
        ],
        p=p,
    )


def image_classification(
    all_p: float = 1.0,
    flip_p: float = 0.5,
    blur_p: float = 0.5,
    brightness_p: float = 0.5,
    noise_p: float = 0.5,
    distort_p: float = 0.0,
    nature_p: float = 0.0,
    resize_p: float = 1.0,
    height: int = 160,
    width: int = 160,
    train: bool = True,
):
    """Augmentation pipeline for image classification models
    Args:
        all_p: probability of applying all list of transforms
        flip_p: probability of applying selected transform in aug_flip
        blur_p: probability of applying selected transform in aug_blur
        brightness_p: probability of applying selected transform in aug_brightness
        noise_p: probability of applying selected transform in aug_noise
        distort_p: probability of applying selected transform in aug_distortion
        nature_p: probability of applying selected transform in aug_nature
        resize_p: probability of applying resize
        height: desired height of an output image
        width: desired width of an output image
        train: Set to True if training mode
    Return:
        aug: Composed list of transformation
    NOTE: Augmentation can be turned off if you
    """
    # no transforamtion for validation and test data
    if not train:
        aug = A.Compose([A.Resize(height=height, width=width, p=resize_p)])
    else:
        aug = A.Compose(
            [
                aug_flip(p=flip_p),
                aug_blur(p=blur_p),
                aug_brightness(p=brightness_p),
                aug_noise(p=noise_p),
                aug_distortion(p=distort_p),
                aug_nature(p=nature_p),
                A.Resize(height=height, width=width, p=resize_p),
            ],
            p=all_p,
        )
    return aug
