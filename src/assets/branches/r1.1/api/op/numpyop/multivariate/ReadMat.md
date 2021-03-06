## ReadMat<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/multivariate/read_mat.py/#L25-L50>View source on Github</a>
```python
ReadMat(
	file: str,
	keys: Union[str, Iterable[str]],
	mode: Union[NoneType, str, Iterable[str]]=None,
	parent_path: str=''
)
```
A class for reading .mat files from disk.

This expects every sample to have a separate .mat file.


<h3>Args:</h3>


* **file**: Dictionary key that contains the .mat path.

* **keys**: Key(s) to read from the .mat file.

* **mode**: What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument like "!infer" or "!train".

* **parent_path**: Parent path that will be prepended to a given filepath.

