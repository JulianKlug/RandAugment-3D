# RandAugment-3D
Implementation of RandAugment in 3D


Original paper: arXiv:1909.13719

Original implementations (in tensorflow):
- From official paper: https://github.com/tensorflow/tpu/blob/298d1fa98638f302ab9df34d9d26bbded7220e8b/models/official/efficientnet/autoaugment.py
- More recent: https://github.com/tensorflow/models/blob/1c79ece9f43340e9bc9571e06a4bf9bd8db8d97a/official/vision/beta/ops/augment.py

Most functions used here are implemented by [MONAI](https://docs.monai.io/en/latest/index.html).

## Installation

`pip install RandAugment3d`

## How to use

Args:
- **_n_**: number of augmentations to apply
- **_magnitude_**: magnitude of augmentations to apply, on a scale where 10 is full level (this scale is used to be as similar as possible to the original implementation). Values > 10 result in more distortion, values < 10 in less.
- **_excluded_operations_**: list of names of the excluded operations. Valid names are listed above.

operations names: ['identity', 'rotate_x', 'rotate_y', 'rotate_z', 'translate', 'scale', 'shear_x', 'shear_y',
                'shear_z', 'shiftIntensity', 'equalize', 'solarize', 'histogramShift', 'sharpen', 'adjustContrast']


Augmentations are performed on volumes with channels last: [n, x, y, z, c]

```
from RandAugment3d import RandAugment3D

augment = RandAugment3D(n=2, magnitude=10)
```


#### Geometric Augmentations only

```
augment = RandAugment3D(n=2, magnitude=10, excluded_operations=['shiftIntensity',
                                                  'equalize',
                                                  'solarize',
                                                  'histogramShift',
                                                  'sharpen',
                                                  'adjustContrast'])
```