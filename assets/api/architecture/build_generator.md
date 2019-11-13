

### build_generator
```python
build_generator(input_shape=(256, 256, 3), num_blocks=9)
```
Returns the generator of the GAN.

#### Args:

* **input_shape (tuple, optional)** :  shape of the input image. Defaults to (256, 256, 3).
* **num_blocks (int, optional)** :  number of resblocks for the generator. Defaults to 9.

#### Returns:

* **'Model' object** :  GAN generator.