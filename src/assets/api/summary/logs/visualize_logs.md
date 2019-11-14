

### visualize_logs
```python
visualize_logs(experiments, save_path=None, smooth_factor=0, share_legend=True, pretty_names=False, ignore_metrics=None, include_metrics=None)
```
A function which will save or display experiment histories for comparison viewing / analysis

#### Args:

* **experiments (list, Experiment)** :  Experiment(s) to plot
* **save_path (str)** :  The path where the figure should be saved, or None to display the figure to the screen
* **smooth_factor (float)** :  A non-negative float representing the magnitude of gaussian smoothing to apply (zero for    none)
* **share_legend (bool)** :  Whether to have one legend across all graphs (true) or one legend per graph (false)
* **pretty_names (bool)** :  Whether to modify the metric names in graph titles (true) or leave them alone (false)
* **ignore_metrics (set)** :  Any metrics to ignore during plotting
* **include_metrics (set)** :  A whitelist of metric keys (None whitelists all keys)

#### Returns:
    The handle of the pyplot figure