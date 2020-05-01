## Gather
```python
Gather(inputs:Union[str, List[str]], outputs:Union[str, List[str]], indices:Union[NoneType, str, List[str]]=None, mode:Union[NoneType, str, Iterable[str]]='eval')
```
Gather values from an input tensor.    If indices are not provided, the maximum values along the batch dimension will be collected. 

#### Args:

* **inputs** :  The tensor(s) to gather values from.
* **indices** :  A tensor containing target indices to gather.
* **outputs** :  The key(s) under which to save the output.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    