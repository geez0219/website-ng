## Augmentation2D
```python
Augmentation2D(inputs=None, outputs=None, mode=None, rotation_range=0.0, width_shift_range=0.0, height_shift_range=0.0, shear_range=0.0, zoom_range=1.0, flip_left_right=False, flip_up_down=False)
```
 This class supports commonly used 2D random affine transformations for data augmentation.    Either a scalar ``x`` or a tuple ``[x1, x2]`` can be specified for rotation, shearing, shifting, and zoom.

#### Args:

* **rotation_range** :  can be one of the following, valid value in [0.0, 360.0)            * Float (x) that represents the range of random rotation (in degrees) from -x to x.            * Tuple of floats ([x1, x2]) that represents  the range of random rotation(in degrees) between x1 and x2.
* **width_shift_range** :  can be one of the following, with the value in [0.0, 1.0)            * Float (x) that represents the range of random width shift (in ratio) between -x and x            * Tuple of floats ([x1, x2]) that represents  the range of random width shift (in ratio) between x1 and x2.
* **height_shift_range** :  can be one of the following, with the value in [0.0, 1.0)            * Float (x) that represents the range of random height shift (in ratio) between -x and x            * Tuple of floats ([x1, x2]) that represents  the range of random height shift (in ratio) between x1 and x2.
* **shear_range** :  can be one of the following, with the value in [0.0, 180)            * Float (x) that represents the range of random shear (in degrees) from -x to x            * Tuple of floats ([x1, x2]) that represents  the range of random shear between x1 and x2.
* **zoom_range** :  can be one of the following, valid value should be non-negative, shrink if < 1.0, zoom if > 1.0.            * Float (x) that represents the range of random zoom/shrink from [min(1.0, x) , max(1.0, x))            * Tuple of floats ([x1, x2]) that represents  the range of random zoom between x1 and x2.
* **flip_left_right** :  Boolean representing whether to flip the image horizontally with a probability of 0.5.
* **flip_up_down** :  Boolean representing whether to flip the image vertically with a probability of 0.5.
* **mode** :  Augmentation on 'training' data or 'evaluation' data.   

### flip
```python
flip(self)
```
        Decides whether or not to flip

#### Returns:
            A boolean that represents whether or not to flip        

### forward
```python
forward(self, data, state)
```
Transforms the data with the augmentation transformation

#### Args:

* **data** :  Data to be transformed
* **state** :  Information about the current execution context

#### Returns:
            Transformed (augmented) data        

### rotate
```python
rotate(self)
```
        Creates affine transformation matrix for 2D rotation

#### Returns:
            Transform affine tensor        

### setup
```python
setup(self)
```
        This method set the appropriate variables necessary for the random 2D augmentation. It also computes the        transformation matrix.

#### Returns:
            None        

### shear
```python
shear(self)
```
        Creates affine transformation matrix for 2D shear

#### Returns:
            Transform affine tensor        

### shift
```python
shift(self)
```
        Creates affine transformation matrix for 2D shift

#### Returns:
            Transform affine tensor        

### transform_matrix_offset_center
```python
transform_matrix_offset_center(self, matrix)
```
        Offsets the tensor to the center of the image

#### Args:

* **matrix** :  Affine tensor

#### Returns:
            An affine tensor offset to the center of the image        

### zoom
```python
zoom(self)
```
        Creates affine transformation matrix for 2D zoom / scale

#### Returns:
            Transform affine tensor        