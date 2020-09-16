## RestoreWizard
```python
RestoreWizard(*args, **kwargs)
```
A trace that can backup and load your entire training status.

System includes model weights, optimizer state, global step and epoch index.


#### Args:

* **directory** :  Directory to save and load system.
* **frequency** :  Saving frequency in epoch(s).