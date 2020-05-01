## EarlyStopping
```python
EarlyStopping(monitor:str='loss', min_delta:float=0.0, patience:int=0, compare:str='min', baseline:Union[float, NoneType]=None, mode:str='eval') -> None
```
Stop training when a monitored quantity has stopped improving.

#### Args:

* **monitor** :  Quantity to be monitored.
* **min_delta** :  Minimum change in the monitored quantity to qualify as an improvement, i.e. an            absolute change of less than min_delta will count as no improvement.
* **patience** :  Number of epochs with no improvement after which training will be stopped.
* **compare** :  One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored            has stopped decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing.
* **baseline** :  Baseline value for the monitored quantity. Training will stop if the model doesn't            show improvement over the baseline.
* **mode** :  What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".

#### Raises:

* **ValueError** :  If `compare` is an invalid value or more than one `monitor` is provided.    