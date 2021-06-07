## RestoreWizard<span class="tag">class</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/restore_wizard.py/#L27-L109>View source on Github</a>
```python
RestoreWizard(
	directory: str,
	frequency: int=1
)
-> None
```
A trace that can backup and load your entire training status.


<h3>Args:</h3>


* **directory**: Directory to save and load the training status.

* **frequency**: Saving frequency in epoch(s).

---

### should_restore<span class="tag">method of RestoreWizard</span><a class="sourcelink" href=https://github.com/fastestimator/fastestimator/blob/r1.1/fastestimator/trace/io/restore_wizard.py/#L69-L75>View source on Github</a>
```python
should_restore(
	self
)
-> bool
```
Whether a restore will be performed.


<h4>Returns:</h4>

<ul class="return-block"><li>    True iff the wizard will perform a restore.</li></ul>

