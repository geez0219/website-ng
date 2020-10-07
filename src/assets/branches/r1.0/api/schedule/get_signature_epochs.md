## get_signature_epochs<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/schedule/schedule.py/#L144-L162>View source on Github</a>
```python
get_signature_epochs(
	items: List[Any],
	total_epochs: int,
	mode: Union[str, NoneType]=None
)
-> List[int]
```
Find all epochs of changes due to schedulers.


<h3>Args:</h3>

* **items** :  List of items to scan from.
* **total_epochs** :  The maximum epoch number to consider when searching for signature epochs.
* **mode** :  Current execution mode. If None, all execution modes will be considered.

<h3>Returns:</h3>
    The epoch numbers of changes.

