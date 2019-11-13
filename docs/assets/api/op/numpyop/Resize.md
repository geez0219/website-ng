## Resize
```python
Resize(target_size, resize_method='bilinear', keep_ratio=False, inputs=None, outputs=None, mode=None)
```
Resize image.

#### Args:

* **target_size (tuple)** :  Target image size in (height, width) format.
* **resize_method (string)** :  `bilinear`, `nearest`, `area`, and `lanczos4` are available.
* **keep_ratio (bool)** :  If `True`, the resulting image will be padded to keep the original aspect ratio.

#### Returns:
    Resized `np.ndarray`.