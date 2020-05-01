## FGSM
```python
FGSM(data:str, loss:str, outputs:str, epsilon:float=0.01, clip_low:Union[float, NoneType]=None, clip_high:Union[float, NoneType]=None, mode:Union[NoneType, str, Iterable[str]]=None)
```
Create an adversarial sample from input data using the Fast Gradient Sign Method.
* **See https** : //arxiv.org/abs/1412.6572 for an explanation of adversarial attacks.

#### Args:

* **data** :  Key of the input to be attacked.
* **loss** :  Key of the loss value to use for gradient computation.
* **outputs** :  The key under which to save the output.
* **epsilon** :  The strength of the perturbation to use in the attack.
* **clip_low** :  a minimum value to clip the output by (defaults to min value of data when set to None).
* **clip_high** :  a maximum value to clip the output by (defaults to max value of data when set to None).
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument            like "!infer" or "!train".    