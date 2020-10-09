## WordtoId<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/op/numpyop/univariate/word_to_id.py/#L22-L65>View source on Github</a>
```python
WordtoId(
	mapping: Union[Dict[str, int], Callable[[List[str]], List[int]]],
	inputs: Union[str, Iterable[str], Callable],
	outputs: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
Converts words to their corresponding id using mapper function or dictionary.


<h3>Args:</h3>


* **mapping**: Mapper function or dictionary

* **inputs**: Key(s) of sequences to be converted to ids.

* **outputs**: Key(s) of sequences are converted to ids.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

