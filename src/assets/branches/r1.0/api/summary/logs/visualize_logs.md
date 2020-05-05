

### visualize_logs
```python
visualize_logs(experiments:List[fastestimator.summary.summary.Summary], save_path:str=None, smooth_factor:float=0, share_legend:bool=True, pretty_names:bool=False, ignore_metrics:Union[Set[str], NoneType]=None, include_metrics:Union[Set[str], NoneType]=None)
```
A function which will save or display experiment histories for comparison viewing / analysis.

#### Args:

* **experiments** :  Experiment(s) to plot.
* **save_path** :  The path where the figure should be saved, or None to display the figure to the screen.
* **smooth_factor** :  A non-negative float representing the magnitude of gaussian smoothing to apply (zero for none).
* **share_legend** :  Whether to have one legend across all graphs (True) or one legend per graph (False).
* **pretty_names** :  Whether to modify the metric names in graph titles (True) or leave them alone (False).
* **ignore_metrics** :  Any metrics to ignore during plotting.
* **include_metrics** :  A whitelist of metric keys (None whitelists all keys).