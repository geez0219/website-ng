## PadSequence
```python
PadSequence(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], max_len:int, value:Union[str, int]=0, append:bool=True, mode:Union[NoneType, str, Iterable[str]]=None) -> None
```
Pad sequences to the same length with provided value.

#### Args:

* **inputs** :  Key(s) of sequences to be padded.
* **outputs** :  Key(s) of sequences that are padded.
* **max_len** :  Maximum length of all sequences.
* **value** :  Padding value.
* **append** :  Pad before or after the sequences. True for padding the values after the sequence, False otherwise.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    