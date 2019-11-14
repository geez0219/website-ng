## MonitorLoss
```python
MonitorLoss()
```
Records loss value. Please don't add this trace into an estimator manually. An estimator will add itautomatically.

### on_batch_begin
```python
on_batch_begin(self, state)
```
Runs at the beginning of every batch of the mode.

#### Args:

* **state (ChainMap)** :  dictionary of run time that has the following key(s)
 * "mode" (str) :  current run time mode, can be "train", "eval" or "test"
 * "epoch" (int) :  current epoch index starting from 0
 * "train_step" (int) :  current global training step starting from 0
 * "batch_idx" (int) :  current local step of the epoch starting from 0
 * "batch_size" (int) :  current global batch size
 * "local_batch_size" (int) :  current batch size for single device                * any keys written by 'on_batch_begin' of previous traces        

### on_begin
```python
on_begin(self, state)
```
Runs once at the beginning of training

#### Args:

* **state (ChainMap)** :  dictionary of run time that has the following key(s)
 * "train_step" (int) :  current global training step starting from 0
 * "num_devices" (int) :  number of devices(mainly gpu) that are being used, if cpu only, the number is 1
 * "log_steps" (int) :  how many training steps between logging intervals
 * "persist_summary" (bool) :  whether to persist the experiment history/summary
 * "total_epochs" (int) :  how many epochs the training is scheduled to run for
 * "total_train_steps" (int) :  how many training steps the training is scheduled to run for                * any keys written by 'on_begin' of previous traces        

### on_end
```python
on_end(self, state)
```
Runs once at the end training. Anything written into the state dictionary will be logged

#### Args:

* **state (ChainMap)** :  dictionary of run time that has the following key(s)
 * "train_step" (int) :  current global training step starting from 0
 * "epoch" (int) :  current epoch index starting from 0
 * "summary" (Experiment) :  will be returned from estimator.fit() if a summary input was specified                * any keys written by 'on_end' of previous traces        