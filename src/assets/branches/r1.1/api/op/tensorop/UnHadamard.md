## UnHadamard<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/un_hadamard.py/#L34-L87>View source on Github</a>
```python
UnHadamard(
	inputs: Union[str, List[str]],
	outputs: Union[str, List[str]],
	n_classes: int,
	code_length: Union[int, NoneType]=None,
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
Convert hadamard encoded class representations into onehot probabilities.


<h3>Args:</h3>

* **inputs** :  Key of the input tensor(s) to be converted.
* **outputs** :  Key of the output tensor(s) as class probabilities.
* **n_classes** :  How many classes are there in the inputs.
* **code_length** :  What code length to use. Will default to the smallest power of 2 which is >= the number of classes.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".



