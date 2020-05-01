## ImageViewer
```python
ImageViewer(inputs:Union[str, Sequence[str]], mode:Union[str, Set[str]]=('eval', 'test')) -> None
```
A trace that interrupts your training in order to display images on the screen.

#### Args:

* **inputs** :  Key(s) of images to be saved.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    