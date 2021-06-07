## GoldenSection<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/search/golden_section.py/#L22-L97>View source on Github</a>
```python
GoldenSection(
	score_fn: Callable[[int, Union[int, float]], float],
	x_min: Union[int, float],
	x_max: Union[int, float],
	max_iter: int,
	integer: bool=True,
	best_mode: str='max',
	name: str='golden_section_search'
)
```
A search class that performs the golden-section search on a single variable.

Golden-section search is good at finding minimal or maximal values of a unimodal function. Each search step reduces
the search range by a constant factor: the golden ratio. More details are available at:
https://en.wikipedia.org/wiki/Golden-section_search.

```python
search = GoldenSection(score_fn=lambda search_idx, n: (n - 3)**2, x_min=0, x_max=6, max_iter=10, best_mode="min")
search.fit()
print(search.get_best_parameters()) # {"n": 3, "search_idx": 2}
```


<h3>Args:</h3>


* **score_fn**: Objective function that measures search fitness. One of its arguments must be 'search_idx' which will be automatically provided by the search routine. This can help with file saving / logging during the search. The other argument should be the variable to be searched over.

* **x_min**: Lower limit (inclusive) of the search space.

* **x_max**: Upper limit (inclusive) of the search space.

* **max_iter**: Maximum number of iterations to run. The range at a given iteration i is 0.618**i * (x_max - x_min). Note that the scoring function will always be evaluated twice before any iterations begin.

* **integer**: Whether the optimized variable is a discrete integer.

* **best_mode**: Whether maximal or minimal fitness is desired. Must be either 'min' or 'max'.

* **name**: The name of the search instance. This is used for saving and loading purposes. 

<h3>Raises:</h3>


* **AssertionError**: If `score_fn`, `x_min`, `x_max`, or `max_iter` are invalid.

