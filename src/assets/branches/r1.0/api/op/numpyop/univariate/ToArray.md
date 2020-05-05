## ToArray
```python
ToArray(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], mode:Union[NoneType, str, Iterable[str]]=None, dtype:Union[str, NoneType]=None)
```
Convert data to a numpy array.

#### Args:

* **inputs** :  Key(s) of the data to be converted.
* **outputs** :  Key(s) into which to write the converted data.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".
* **dtype** :  The dtype to apply to the output array, or None to infer the type.    