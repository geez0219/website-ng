## plot_logs<span class="tag">function</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/summary/logs/log_plot.py/#L149-L367>View source on Github</a>
```python
plot_logs(
	experiments: List[fastestimator.summary.summary.Summary],
	smooth_factor: float=0,
	share_legend: bool=True,
	ignore_metrics: Union[Set[str], NoneType]=None,
	pretty_names: bool=False,
	include_metrics: Union[Set[str], NoneType]=None
)
-> matplotlib.figure.Figure
```
A function which will plot experiment histories for comparison viewing / analysis.


<h3>Args:</h3>


* **experiments**: Experiment(s) to plot.

* **smooth_factor**: A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).

* **share_legend**: Whether to have one legend across all graphs (True) or one legend per graph (False).

* **pretty_names**: Whether to modify the metric names in graph titles (True) or leave them alone (False).

* **ignore_metrics**: Any keys to ignore during plotting.

* **include_metrics**: A whitelist of keys to include during plotting. If None then all will be included. 

<h3>Returns:</h3>

<ul class="return-block"><li>    The handle of the pyplot figure.</li></ul>

