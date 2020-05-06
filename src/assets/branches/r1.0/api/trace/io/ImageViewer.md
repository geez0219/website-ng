## ImageViewer
```python
ImageViewer(inputs:Union[str, Sequence[str]], mode:Union[str, Set[str]]=('eval', 'test'), width:int=12, height:int=6) -> None
```
A trace that interrupts your training in order to display images on the screen.

This class is useful primarily for Jupyter Notebook, or for debugging purposes.



#### Args:

* **inputs** :  Key(s) of images to be displayed.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **width** :  The width in inches of the figure.
* **height** :  The height in inches of the figure.    