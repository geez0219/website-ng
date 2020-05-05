## Argmax
```python
Argmax(inputs:Union[str, List[str]], outputs:Union[str, List[str]], axis:int=0, mode:Union[NoneType, str, Iterable[str]]='eval')
```
Get the argmax from a tensor.

#### Args:

* **inputs** :  The tensor(s) to gather values from.
* **outputs** :  The key(s) under which to save the output.
* **axis** :  The axis along which to collect the argmax.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    