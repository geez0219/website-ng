

### plot_logs
```python
plot_logs(experiments, smooth_factor=0, share_legend=True, ignore_metrics=None, pretty_names=False, include_metrics=None)
```
A function which will plot experiment histories for comparison viewing / analysis

#### Args:

* **experiments (list, Experiment)** :  Experiment(s) to plot
* **smooth_factor (float)** :  A non-negative float representing the magnitude of gaussian smoothing to apply (zero for    none)
* **share_legend (bool)** :  Whether to have one legend across all graphs (true) or one legend per graph (false)
* **pretty_names (bool)** :  Whether to modify the metric names in graph titles (true) or leave them alone (false)
* **ignore_metrics (set)** :  Any keys to ignore during plotting
* **include_metrics (set)** :  A whitelist of keys to include during plotting. If None then all will be included.

#### Returns:
    The handle of the pyplot figure