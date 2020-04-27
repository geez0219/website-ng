## ResizeImageAndBbox
```python
ResizeImageAndBbox(target_size, resize_method='bilinear', keep_ratio=False, inputs=None, outputs=None, mode=None)
```
Resize image and associated bounding boxes for object detection.

#### Args:

* **target_size (tuple)** :  Target image size in (height, width) format.
* **resize_method (string)** :  `bilinear`, `nearest`, `area`, and `lanczos4` are available.
* **keep_ratio (bool)** :  If `True`, the resulting image will be padded to keep the original aspect ratio.
* **inputs (list, optional)** :  This list of 5 strings has to be in the order of image, x1 coordinates, y1            coordinates, widths, and heights. For example, `['img', 'x1', 'y1', 'width', 'height']`.
* **outputs (list, optional)** :  Output feature names.
* **mode (str, optional)** :  Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always            execute. Defaults to 'eval'.

#### Returns:

* **list** :  `[resized_image, x1, y1, widths, heights]`.    