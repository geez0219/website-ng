## ImageOnlyAlbumentation
```python
ImageOnlyAlbumentation(func:albumentations.core.transforms_interface.ImageOnlyTransform, inputs:Union[str, List[str], Callable], outputs:Union[str, List[str]], mode:Union[NoneType, str, Iterable[str]]=None)
```
Operators which apply to single images (as opposed to images + masks or images + bounding boxes).
* **This is a wrapper for functionality provided by the Albumentations library** : 
* **https** : //github.com/albumentations-team/albumentations. A useful visualization tool for many of the possible effects
* **it provides is available at https** : //albumentations-demo.herokuapp.com.

#### Args:

* **func** :  An Albumentation function to be invoked.
* **inputs** :  Key(s) from which to retrieve data from the data dictionary. If more than one key is provided, the            `func` will be run in replay mode so that the exact same augmentation is applied to each value.
* **outputs** :  Key(s) under which to write the outputs of this Op back to the data dictionary.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    