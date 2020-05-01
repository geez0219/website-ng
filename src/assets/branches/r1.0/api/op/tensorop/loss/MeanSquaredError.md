## MeanSquaredError
```python
MeanSquaredError(inputs:Union[NoneType, str, Iterable[str], Callable]=None, outputs:Union[NoneType, str, Iterable[str]]=None, mode:Union[NoneType, str, Iterable[str]]=None, average_loss:bool=True)
```
Calculate the mean squared error loss between two tensors.

#### Args:

* **inputs** :  A tuple or list like [<y_true>, <y_pred>].
* **outputs** :  String key under which to store the computed loss.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    