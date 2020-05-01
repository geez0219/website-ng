## ReadImage
```python
ReadImage(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, parent_path:str='', grey_scale:bool=False)
```
A class for reading png or jpg images from disk.

#### Args:

* **inputs** :  Key(s) of paths to images to be loaded.
* **outputs** :  Key(s) of images to be output.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **parent_path** :  Parent path that will be prepended to a given path.
* **grey_scale** :  Whether or not to read the image as grayscale.    