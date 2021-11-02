import random
import numpy as np
import tensorflow as tf
from skimage.exposure import equalize_hist
from monai.transforms import RandGaussianNoise, RandStdShiftIntensity, RandScaleIntensity, RandAdjustContrast, \
    RandGaussianSharpen, RandHistogramShift, RandAffine, Rand3DElastic


def solarize(image: tf.Tensor, threshold: int = 0.5) -> tf.Tensor:
    """Solarize the input image(s)."""
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract max from the pixel.
    return tf.where(image < threshold * tf.reduce_max(image), image, tf.reduce_max(image) - image).numpy()


def equalize(volume, magnitude=10):
    return equalize_hist(volume)


def elastic(volume, magnitude=10):
    transformer = Rand3DElastic(sigma_range=(5, 7), magnitude_range=(100, 150), prob=1, padding_mode='zeros')
    return transformer(volume)


def rotate_3d(volume, magnitude=10):
    # full level in radians (approx 30°)
    max_level = 0.5
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, rotate_range=((level, -level), (level, -level),
                                                   (level, -level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def rotate_x(volume, magnitude=10):
    # rotation along x axis
    # full level in radians (approx 30°)
    max_level = 0.5
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, rotate_range=((level, -level), (0), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def rotate_y(volume, magnitude=10):
    # rotation along y axis
    # full level in radians (approx 30°)
    max_level = 0.5
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, rotate_range=((0), (level, -level), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def rotate_z(volume, magnitude=10):
    # rotation along z axis
    # full level in radians (approx 30°)
    max_level = 0.5
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, rotate_range=((0), (0), (level, -level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def shear_3d(volume, magnitude=10):
    max_level = 0.3
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, shear_range=(
        (level, -level), (level, -level), (level, -level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def shear_x(volume, magnitude=10):
    max_level = 0.3
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, shear_range=((level, -level), (0), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def shear_y(volume, magnitude=10):
    max_level = 0.3
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, shear_range=((0), (level, -level), (0)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def shear_z(volume, magnitude=10):
    max_level = 0.3
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, shear_range=((0), (0), (level, -level)), padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def translate(volume, magnitude=10):
    max_level = 10
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, translate_range=((-level, level), (-level, level), (-level, level)),
                             padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def scale(volume, magnitude=10):
    max_level = 0.3
    level = (magnitude / 10) * max_level
    transformer = RandAffine(prob=1, scale_range=level, padding_mode='zeros')
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def noise(volume, magnitude=10):
    transformer = RandGaussianNoise()
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def shiftIntensity(volume, magnitude=10):
    max_level = 1
    level = (magnitude / 10) * max_level
    transformer = RandStdShiftIntensity(prob=1, factors=(-level, level))
    return transformer(volume)


def scaleIntensity(volume, magnitude=10):
    transformer = RandScaleIntensity(prob=1, factors=(5, 10))
    transformed_volume = transformer(volume)
    return safe_numpy_conversion(transformed_volume)


def adjustContrast(volume, magnitude=10):
    max_level = 1.9
    level = (magnitude / 10) * max_level
    transformer = RandAdjustContrast(prob=1, gamma=(0.5, level))

    return transformer(volume)


def sharpen(volume, magnitude=10):
    max_level = 1.9
    level = (magnitude / 10) * max_level
    sigma_upper_bound = 0.5
    sigma_lower_bound = 1
    transformer = RandGaussianSharpen(sigma1_x=(sigma_upper_bound, sigma_lower_bound),
                                      sigma1_y=(sigma_upper_bound, sigma_lower_bound),
                                      sigma1_z=(sigma_upper_bound, sigma_lower_bound), sigma2_x=0.5,
                                      sigma2_y=0.5, sigma2_z=0.5, alpha=(level, level), prob=1)
    return transformer(volume)


def histogramShift(volume, magnitude=10):
    max_level = 10
    level = (magnitude / 10) * max_level
    transformer = RandHistogramShift(num_control_points=[level, 2 * level], prob=1)
    return transformer(volume)


def identity(volume, magnitude=10):
    return volume


def to_channels_first(tensor):
    return tf.experimental.numpy.moveaxis(tensor, 3, 0)


def to_channels_last(tensor):
    return tf.experimental.numpy.moveaxis(tensor, 0, 3)


def safe_numpy_conversion(object):
    if type(object) is np.ndarray:
        return object
    else:
        return object.numpy()


def augment_list():
    l = [
        identity,
        # elastic,
        rotate_x,
        rotate_y,
        rotate_z,
        translate,
        scale,
        shear_x,
        shear_y,
        shear_z,
        shiftIntensity,
        equalize,
        solarize,
        histogramShift,
        sharpen,
        adjustContrast,
    ]

    return l


'''
Transforms in original paper: [
    ’Identity’, ’AutoContrast’, ’Equalize’,
    ’Rotate’, ’Solarize’, ’Color’, ’Posterize’,
    ’Contrast’, ’Brightness’, ’Sharpness’,
    ’ShearX’, ’ShearY’, ’TranslateX’, ’TranslateY’]
'''


class RandAugment3D:
    """
    Implementation of RandAugment in 3D.

    operations names: ['identity', 'rotate_x', 'rotate_y', 'rotate_z', 'translate', 'scale', 'shear_x', 'shear_y',
                'shear_z', 'shiftIntensity', 'equalize', 'solarize', 'histogramShift', 'sharpen', 'adjustContrast']

    Original Paper: Original paper: arXiv:1909.13719

    Args:
        n: number of augmentations to apply

        magnitude: magnitude of augmentations to apply, on a scale where 10 is full level (this scale is used to be as similar as possible to the original implementation). Values > 10 result in more distortion, values < 10 in less.

        excluded_operations: list of names of the excluded operations. Valid names are listed above.
    """
    def __init__(self, n: int, magnitude: int = 10, excluded_operations: list = []):
        self.n = n
        self.magnitude = magnitude

        # use all operations except the excluded ones
        if not all(x in [operation.__name__ for operation in augment_list()] for x in excluded_operations):
            raise ValueError(f'One operation in {excluded_operations} is not a valid operation. All valid operations '
                             f'are listed here {[operation.__name__ for operation in augment_list()]}')
        self.augment_list = [operation for operation in augment_list()
                             if operation.__name__ not in excluded_operations]

    def __call__(self, volume):
        volume = to_channels_first(volume).numpy()
        ops = random.sample(self.augment_list, k=self.n)
        for op in ops:
            volume = op(volume, self.magnitude)
        volume = to_channels_last(volume)
        volume = tf.cast(volume, tf.float32)

        return volume
