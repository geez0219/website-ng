## Watch
```python
Watch(inputs:Union[NoneType, str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None) -> None
```
Watch one or more tensors for later gradient computation.

#### Args:

* **inputs** :  which tensors to watch during future computation.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    