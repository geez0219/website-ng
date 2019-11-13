## EarlyStopping
```python
EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, compare='min', baseline=None, restore_best_weights=False, mode='eval')
```
Stop training when a monitored quantity has stopped improving.

#### Args:

* **monitor (str, optional)** :  Quantity to be monitored.. Defaults to "loss".
* **min_delta (int, optional)** :  Minimum change in the monitored quantity to qualify as an improvement, i.e. an        absolute change of less than min_delta, will count as no improvement. Defaults to 0.
* **patience (int, optional)** :  Number of epochs with no improvement after which training will be stopped. Defaults        to 0.
* **verbose (int, optional)** :  Verbosity mode.. Defaults to 0.
* **compare (str, optional)** :  One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored        has stopped decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing.        Defaults to 'min'.
* **baseline (float, optional)** :  Baseline value for the monitored quantity. Training will stop if the model doesn't        show improvement over the baseline. Defaults to None.
* **restore_best_weights (bool, optional)** :  Whether to restore model weights from the epoch with the best value of        the monitored quantity. If False, the model weights obtained at the last step of training are used.        Defaults to False.
* **mode (str, optional)** :  Restrict the trace to run only on given modes {'train', 'eval', 'test'}. None will always                execute. Defaults to 'eval'.