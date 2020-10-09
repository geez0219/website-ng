## RestoreWizard<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/restore_wizard.py/#L28-L96>View source on Github</a>
```python
RestoreWizard(
	directory: str,
	frequency: int=1
)
-> None
```
A trace that can backup and load your entire training status.

System includes model weights, optimizer state, global step and epoch index.


<h3>Args:</h3>


* **directory**: Directory to save and load system.

* **frequency**: Saving frequency in epoch(s).

