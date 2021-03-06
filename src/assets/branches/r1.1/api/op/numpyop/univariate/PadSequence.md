## PadSequence<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/univariate/pad_sequence.py/#L24-L71>View source on Github</a>
```python
PadSequence(
	inputs: Union[str, Iterable[str]],
	outputs: Union[str, Iterable[str]],
	max_len: int,
	value: Union[str, int]=0,
	append: bool=True,
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
Pad sequences to the same length with provided value.


<h3>Args:</h3>


* **inputs**: Key(s) of sequences to be padded.

* **outputs**: Key(s) of sequences that are padded.

* **max_len**: Maximum length of all sequences.

* **value**: Padding value.

* **append**: Pad before or after the sequences. True for padding the values after the sequence, False otherwise.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

