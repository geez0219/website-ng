## SuperLoss<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/op/tensorop/loss/super_loss.py/#L33-L169>View source on Github</a>
```python
SuperLoss(
	loss: fastestimator.op.tensorop.loss.loss.LossOp,
	threshold: Union[float, str]='exp',
	regularization: float=1.0,
	average_loss: bool=True,
	output_confidence: Union[str, NoneType]=None
)
```
Loss class to compute a 'super loss' (automatic curriculum learning) based on a regular loss.

This class adds automatic curriculum learning on top of any other loss metric. It is especially useful in for noisy
datasets. See https://papers.nips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf for details.


<h3>Args:</h3>


* **loss**: A loss object which we use to calculate the underlying regular loss. This should be an object of type fe.op.tensorop.loss.loss.LossOp.

* **threshold**: Either a constant value corresponding to an average expected loss (for example log(n_classes) for cross-entropy classification), or 'exp' to use an exponential moving average loss.

* **regularization**: The regularization parameter to use for the super loss (must by >0, as regularization approaches infinity the SuperLoss converges to the regular loss value).

* **average_loss**: Whether the final loss should be averaged or not.

* **output_confidence**: If not None then the confidence scores for each sample will be written into the specified key. This can be useful for finding difficult or mislabeled data. 

<h3>Raises:</h3>


* **ValueError**: If the provided `loss` has multiple outputs or the `regularization` / `threshold` parameters are
        invalid.

