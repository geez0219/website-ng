## Onehot
```python
Onehot(inputs:Union[str, Iterable[str], Callable], outputs:Union[str, Iterable[str]], num_classes:int, label_smoothing:float=0.0, mode:Union[NoneType, str, Iterable[str]]=None)
```
Transform an integer label to one-hot-encoding.
* **This can be desirable for increasing robustness against incorrect labels** : 
* **https** : //towardsdatascience.com/label-smoothing-making-model-robust-to-incorrect-labels-2fae037ffbd0

#### Args:

* **inputs** :  Input key(s) of labels to be onehot encoded.
* **outputs** :  Output key(s) of labels.
* **num_classes** :  Total number of classes.
* **label_smoothing** :  Smoothing factor, after smoothing class output is 1 - label_smoothing + label_smoothing
* **/ num_classes, the other class output is** :  label_smoothing / num_classes.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    