## ImageSaver
```python
ImageSaver(inputs:Union[str, Sequence[str]], save_dir:str='/home/ubuntu/vivek', dpi:int=300, mode:Union[str, Set[str]]=('eval', 'test')) -> None
```
A trace that saves images to the disk.

#### Args:

* **inputs** :  Key(s) of images to be saved.
* **save_dir** :  The directory into which to write the images.
* **dpi** :  How many dots per inch to save.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    