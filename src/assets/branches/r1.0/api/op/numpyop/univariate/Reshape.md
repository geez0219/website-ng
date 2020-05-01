## Reshape
```python
Reshape(shape:Union[int, Tuple[int, ...]], inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None)
```
An Op which re-shapes data to a target shape.

#### Args:

* **shape** :  The desired output shape. At most one value may be -1 to put all of the leftover elements into that axis.
* **inputs** :  Key(s) of the data to be reshaped.
* **outputs** :  Key(s) into which to write the converted data.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    