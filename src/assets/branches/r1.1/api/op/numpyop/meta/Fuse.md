## Fuse<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/op/numpyop/meta/fuse.py/#L25-L57>View source on Github</a>
```python
Fuse(
	ops: Union[fastestimator.op.numpyop.numpyop.NumpyOp, List[fastestimator.op.numpyop.numpyop.NumpyOp]]
)
-> None
```
Run a sequence of NumpyOps as a single Op.


<h3>Args:</h3>

* **ops** :  A sequence of NumpyOps to run. They must all share the same mode. It also doesn't support scheduled ops at        the moment, though the Fuse itself may be scheduled.

<h3>Raises:</h3>

* **ValueError** :  If `repeat` or `ops` are invalid.



