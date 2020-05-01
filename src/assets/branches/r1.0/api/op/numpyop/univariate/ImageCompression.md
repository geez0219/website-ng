## ImageCompression
```python
ImageCompression(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, quality_lower:float=99, quality_upper:float=100, compression_type:albumentations.augmentations.transforms.ImageCompression.ImageCompressionType=<ImageCompressionType.JPEG: 0>)
```
Decrease compression of an image.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **quality_lower** :  Lower bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.
* **quality_upper** :  Upper bound on the image quality. Should be in [0, 100] range for jpeg and [1, 100] for webp.
* **compression_type** :  should be ImageCompressionType.JPEG or ImageCompressionType.WEBP.
* **Image types** :         uint8, float32    