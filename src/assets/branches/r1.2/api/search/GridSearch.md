## GridSearch<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/grid_search.py/#L22-L60>View source on Github</a>
```python
GridSearch(
	score_fn: Callable[..., float],
	params: Dict[str, List],
	best_mode: str='max',
	name: str='grid_search'
)
```
A class which executes a grid search.

Grid search can be used to find the optimal combination of one or more hyperparameters.

```python
search = GridSearch(score_fn=lambda search_idx, a, b: a + b, params={"a": [1, 2, 3], "b": [4, 5, 6]})
search.fit()
print(search.get_best_parameters()) # {"a": 3, "b": 6, "search_idx": 9}
```


<h3>Args:</h3>


* **score_fn**: Objective function that measures search fitness. One of its arguments must be 'search_idx' which will be automatically provided by the search routine. This can help with file saving / logging during the search.

* **params**: A dictionary with key names matching the `score_fn`'s inputs. Its values should be lists of options.

* **best_mode**: Whether maximal or minimal fitness is desired. Must be either 'min' or 'max'.

* **name**: The name of the search instance. This is used for saving and loading purposes. 

<h3>Raises:</h3>


* **AssertionError**: If `params` is not dictionary, or contains key not used by `score_fn`

