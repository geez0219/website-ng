## SaliencyNet<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/xai/saliency.py/#L43-L261>View source on Github</a>
```python
SaliencyNet(
	model: ~Model,
	model_inputs: Union[str, Sequence[str]],
	model_outputs: Union[str, Sequence[str]],
	outputs: Union[str, List[str]]='saliency'
)
```
A class to generate saliency masks from a given model.


<h3>Args:</h3>


* **model**: The model, compiled with fe.build, which is to be inspected.

* **model_inputs**: The key(s) corresponding to the model inputs within the data dictionary.

* **model_outputs**: The key(s) corresponding to the model outputs which are written into the data dictionary.

* **outputs**: The keys(s) under which to write the generated saliency images.

---

### get_integrated_masks<span class="tag">method of SaliencyNet</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/xai/saliency.py/#L236-L261>View source on Github</a>
```python
get_integrated_masks(
	self,
	batch: Dict[str, Any],
	nsamples: int=25
)
-> Dict[str, Union[~Tensor, numpy.ndarray]]
```
Generates integrated greyscale saliency mask(s) from a given `batch` of data.

See https://arxiv.org/abs/1703.01365 for background on the IntegratedGradient method.


<h4>Args:</h4>


* **batch**: An input batch of data.

* **nsamples**: Number of samples to average across to get the integrated gradient. 

<h4>Returns:</h4>

<ul class="return-block"><li>    Greyscale saliency masks smoothed via the IntegratedGradient method.</li></ul>

---

### get_masks<span class="tag">method of SaliencyNet</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/xai/saliency.py/#L107-L121>View source on Github</a>
```python
get_masks(
	self,
	batch: Dict[str, Any]
)
-> Dict[str, Union[~Tensor, numpy.ndarray]]
```
Generates greyscale saliency mask(s) from a given `batch` of data.


<h4>Args:</h4>


* **batch**: A batch of input data to be fed to the model. 

<h4>Returns:</h4>

<ul class="return-block"><li>    The model's classification decisions and greyscale saliency mask(s) for the given <code>batch</code> of data.</li></ul>

---

### get_smoothed_masks<span class="tag">method of SaliencyNet</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.0/fastestimator/xai/saliency.py/#L184-L234>View source on Github</a>
```python
get_smoothed_masks(
	self,
	batch: Dict[str, Any],
	stdev_spread: float=0.15,
	nsamples: int=25,
	nintegration: Union[int, NoneType]=None,
	magnitude: bool=True
)
-> Dict[str, Union[~Tensor, numpy.ndarray]]
```
Generates smoothed greyscale saliency mask(s) from a given `batch` of data.


<h4>Args:</h4>


* **batch**: An input batch of data.

* **stdev_spread**: Amount of noise to add to the input, as fraction of the total spread (x_max - x_min).

* **nsamples**: Number of samples to average across to get the smooth gradient.

* **nintegration**: Number of samples to compute when integrating (None to disable).

* **magnitude**: If true, computes the sum of squares of gradients instead of just the sum. 

<h4>Returns:</h4>

<ul class="return-block"><li>    Greyscale saliency mask(s) smoothed via the SmoothGrad method.</li></ul>

