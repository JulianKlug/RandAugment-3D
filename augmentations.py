import random
import tensorflow as tf
from skimage.exposure import equalize_hist
from monai.transforms import RandGaussianNoise, RandStdShiftIntensity, RandScaleIntensity, RandAdjustContrast, \
    RandGaussianSharpen, RandHistogramShift, RandAffine, Rand3DElastic

def solarize(image: tf.Tensor, threshold: int = 0.5) -> tf.Tensor:
  """Solarize the input image(s)."""
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract max from the pixel.
  return tf.where(image < threshold*tf.reduce_max(image), image, tf.reduce_max(image) - image).numpy()

def equalize(volume):
    return equalize_hist(volume)

def elastic(volume):
    transformer = Rand3DElastic(sigma_range=(5, 7), magnitude_range=(100, 150), prob=1, padding_mode='zeros')
    return transformer(volume)

def rotate(volume):
    transformer = RandAffine(prob=1, rotate_range=((30, -30), (10, -10), (10, -10)), padding_mode='zeros')
    return transformer(volume).numpy()

def shear(volume):
    transformer = RandAffine(prob=1, shear_range=((1, -1), (1, -1), (1, -1)), padding_mode='zeros')
    return transformer(volume).numpy()

def translate(volume):
    transformer = RandAffine(prob=1, translate_range=((10, -10), (10, -10), (10, -10)), padding_mode='zeros')
    return transformer(volume).numpy()

def scale(volume):
    transformer = RandAffine(prob=1, scale_range=(0.9, 1.1), padding_mode='zeros')
    return transformer(volume).numpy()

def noise(volume):
    transformer = RandGaussianNoise()
    return transformer(volume).numpy()

def shiftIntensity(volume):
    transformer = RandStdShiftIntensity(prob=1, factors=(5, 10))
    return transformer(volume)

def scaleIntensity(volume):
    transformer = RandScaleIntensity(prob=1, factors=(5, 10))
    return transformer(volume).numpy()

def adjustContrast(volume):
    transformer = RandAdjustContrast(prob=1)
    return transformer(volume)

def sharpen(volume):
    transformer = RandGaussianSharpen(prob=1)
    return transformer(volume)

def histogramShift(volume):
    transformer = RandHistogramShift(num_control_points=10, prob=1)
    return transformer(volume)


def identity(volume):
    return volume

def to_channels_first(tensor):
    return tf.experimental.numpy.moveaxis(tensor, 3, 0)

def to_channels_last(tensor):
    return tf.experimental.numpy.moveaxis(tensor, 0, 3)


def augment_list():
    l = [
        identity,
        elastic,
        rotate,
        translate,
        scale,
        shear,
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
    def __init__(self, n):
        self.n = n
        # TODO Implement magnitude
        # self.magnitude = m
        self.augment_list = augment_list()

    def __call__(self, volume):
        volume = to_channels_first(volume).numpy()
        ops = random.sample(self.augment_list, k=self.n)
        for op in ops:
            print(op.__name__)
            volume = op(volume)
        volume = to_channels_last(volume)

        return volume