## Search<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/search.py/#L23-L167>View source on Github</a>
```python
Search(
	score_fn: Callable[..., float],
	best_mode: str='max',
	name: str='search'
)
```
Base class which other searches inherit from.

The base search class takes care of evaluation logging, saving and loading, and is also able to recover from
interrupted search runs and cache the search history.


<h3>Args:</h3>


* **score_fn**: Objective function that measures search fitness. One of its arguments must be 'search_idx' which will be automatically provided by the search routine. This can help with file saving / logging during the search.

* **best_mode**: Whether maximal or minimal fitness is desired. Must be either 'min' or 'max'.

* **name**: The name of the search instance. This is used for saving and loading purposes. 

<h3>Raises:</h3>


* **AssertionError**: If `best_mode` is not 'min' or 'max', or search_idx is not an input argument of `score_fn`.

---

### evaluate<span class="tag">method of Search</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/search.py/#L53-L77>View source on Github</a>
```python
evaluate(
	self, **kwargs: Any
)
-> float
```
Evaluate the score function and return the score.


<h4>Args:</h4>


* **kwargs**: Any keyword argument(s) to pass to the score function. Should not contain search_idx as this will be populated manually here. 

<h4>Returns:</h4>

<ul class="return-block"><li>    Fitness score calculated by <code>score_fn</code>.</li></ul>

---

### fit<span class="tag">method of Search</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/search.py/#L150-L163>View source on Github</a>
```python
fit(
	self,
	save_dir: str=None
)
-> None
```
Start the search.


<h4>Args:</h4>


* **save_dir**: When `save_dir` is provided, the search results will be backed up to the `save_dir` after each evaluation. It will also attempt to load the search state from `save_dir` if possible. This is useful when the search might experience interruption since it can be restarted using the same command.

---

### get_best_results<span class="tag">method of Search</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/search.py/#L79-L94>View source on Github</a>
```python
get_best_results(
	self
)
-> Tuple[Dict[str, Any], float]
```
Get the best result from the current search history.


<h4>Raises:</h4>


* **RuntimeError**: If the search hasn't been run yet.

<h4>Returns:</h4>

<ul class="return-block"><li>    The best results in the format of (parameter, score)

</li></ul>

---

### get_search_results<span class="tag">method of Search</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/search.py/#L96-L102>View source on Github</a>
```python
get_search_results(
	self
)
-> List[Tuple[Dict[str, Any], float]]
```
Get the current search history.


<h4>Returns:</h4>

<ul class="return-block"><li>    The evluation history list, with each element being a tuple of parameters and score.</li></ul>

---

### load<span class="tag">method of Search</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/search.py/#L123-L148>View source on Github</a>
```python
load(
	self,
	load_dir: str,
	not_exist_ok: bool=False
)
-> None
```
Load the state of search from a given directory. It will look for `name.json` within the `load_dir`.


<h4>Args:</h4>


* **load_dir**: The folder path to load the state from.

* **not_exist_ok**: whether to ignore when the file does not exist.

---

### save<span class="tag">method of Search</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/search.py/#L112-L121>View source on Github</a>
```python
save(
	self,
	save_dir: str
)
-> None
```
Save the state of the instance to a specific directory, it will create `name.json` file in the `save_dir`.


<h4>Args:</h4>


* **save_dir**: The folder path to save to.

