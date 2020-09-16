## CutMixBatch
```python
CutMixBatch(inputs:str, outputs:Iterable[str], mode:Union[NoneType, str, Iterable[str]]='train', alpha:Union[float, ~Tensor]=1.0) -> None
```
This class performs cutmix augmentation on a batch of tensors.

In this augmentation technique patches are cut and pasted among training images where the ground truth labels are
also mixed proportionally to the area of the patches. This class should be used in conjunction with MixLoss to
perform CutMix training, which helps to reduce over-fitting, perform object detection, and against adversarial
attacks (https://arxiv.org/pdf/1905.04899.pdf).


#### Args:

* **inputs** :  Key of the image batch to be cut-mixed.
* **outputs** :  Keys under which to store the cut-mixed images and lambda value.
* **mode** :  What mode(s) to execute this Op in. For example, "train", "eval", "test", or "infer". To execute        regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument        like "!infer" or "!train".
* **alpha** :  The alpha value defining the beta distribution to be drawn from during training which controls the        combination ratio between image pairs.

#### Raises:

* **AssertionError** :  If the provided inputs are invalid.