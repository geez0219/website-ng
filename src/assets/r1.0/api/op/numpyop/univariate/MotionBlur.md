## MotionBlur
```python
MotionBlur(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, blur_limit:Union[int, Tuple[int, int]]=7)
```
Motion Blur the image with a randomly-sized kernel.

#### Args:

* **inputs** :  Key(s) of images to be modified.
* **outputs** :  Key(s) into which to write the modified images.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **blur_limit** :  maximum kernel size for blurring the input image. Should be in the range [3, inf).
* **Image types** :         uint8, float32    