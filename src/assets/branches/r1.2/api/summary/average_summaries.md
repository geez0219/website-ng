## average_summaries<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/summary/summary.py/#L94-L169>View source on Github</a>
```python
average_summaries(
	name: str,
	summaries: List[fastestimator.summary.summary.Summary]
)
-> fastestimator.summary.summary.Summary
```
Average multiple summaries together, storing their metric means +- stdevs.


<h3>Args:</h3>


* **name**: The name for the new summary.

* **summaries**: A list of summaries to be averaged. 

<h3>Returns:</h3>

<ul class="return-block"><li>    A single summary object reporting mean+-stddev for each metric. If a particular value has only 1 datapoint, it
    will not be averaged.</li></ul>

