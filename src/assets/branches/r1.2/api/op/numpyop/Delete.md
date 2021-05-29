## Delete<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/numpyop/numpyop.py/#L101-L113>View source on Github</a>
```python
Delete(
	keys: Union[str, List[str]],
	mode: Union[NoneType, str, Iterable[str]]=None
)
-> None
```
Delete key(s) and their associated values from the data dictionary.

The system has special logic to detect instances of this Op and delete its `inputs` from the data dictionary.


<h3>Args:</h3>


* **keys**: Existing key(s) to be deleted from the data dictionary.

