## MixUpBatch<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/tensorop/augmentation/mixup_batch.py/#L29-L76>View source on Github</a>
```python
MixUpBatch(
	inputs: Union[str, Iterable[str]],
	outputs: Iterable[str],
	mode: Union[NoneType, str, Iterable[str]]='train',
	alpha: float=1.0,
	shared_beta: bool=True
)
```
MixUp augmentation for tensors.

This class should be used in conjunction with MixLoss to perform mix-up training, which helps to reduce
over-fitting, stabilize GAN training, and against adversarial attacks (https://arxiv.org/abs/1710.09412).


<h3>Args:</h3>


* **inputs**: Key of the input to be mixed up.

* **outputs**: Key to store the mixed-up outputs.

* **mode**: What mode to execute in. Probably 'train'.

* **alpha**: The alpha value defining the beta distribution to be drawn from during training.

* **shared_beta**: Sample a single beta for a batch or element wise beta for each image. 

<h3>Raises:</h3>


* **AssertionError**: If input arguments are invalid.

