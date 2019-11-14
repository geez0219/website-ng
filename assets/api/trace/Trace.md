## Trace
```python
Trace(inputs=None, outputs=None, mode=None)
```
Trace base class. User can use `Trace` to customize their own operations during training, validation and testing.The `Network` instance can be accessible by `self.network`. Trace execution order will attempt to be inferredwhenever possible based on the provided inputs and outputs variables.

#### Args:

* **inputs (str, list, set)** :  A set of keys that this trace intends to read from the state dictionary as inputs
* **outputs (str, list, set)** :  A set of keys that this trace intends to write into the state dictionary
* **mode (string)** :  Restrict the trace to run only on given modes ('train', 'eval', 'test'). None will always                    execute

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

### on_batch_end
```python
on_batch_end(self, state)
```
Runs at the end of every batch of the mode. Anything written to the top level of the state dictionary will be        printed in the logs. Things written only to the batch sub-dictionary will not be logged

#### Args:

* **state (ChainMap)** :  dictionary of run time that has the following key(s)
 * "mode" (str) :   current run time mode, can be "train", "eval" or "test"
 * "epoch" (int) :  current epoch index starting from 0
 * "train_step" (int) :  current global training step starting from 0
 * "batch_idx" (int) :  current local step of the epoch starting from 0
 * "batch_size" (int) :  current global batch size
 * "batch" (dict) :  the batch data after the Network execution
 * "local_batch_size" (int) :  current batch size for single device
 * <loss_name> defined in model (float) :  loss of current batch (only available when mode is "train")                * any keys written by 'on_batch_end' of previous traces        

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

### on_epoch_begin
```python
on_epoch_begin(self, state)
```
Runs at the beginning of each epoch of the mode.

#### Args:

* **state (ChainMap)** :  dictionary of run time that has the following key(s)
 * "mode" (str) :   current run time mode, can be "train", "eval" or "test"
 * "epoch" (int) :  current epoch index starting from 0
 * "train_step" (int) :  current global training step starting from 0
 * "num_examples" (int) :  total number of examples available for current mode                * any keys written by 'on_epoch_begin' of previous traces        

### on_epoch_end
```python
on_epoch_end(self, state)
```
Runs at the end of every epoch of the mode. Anything written into the state dictionary will be logged

#### Args:

* **state (ChainMap)** :  dictionary of run time that has the following key(s)
 * "mode" (str) :   current run time mode, can be "train", "eval" or "test"
 * "epoch" (int) :  current epoch index starting from 0
 * "train_step" (int) :  current global training step starting from 0
 * <loss_name> defined in model (float) :  average loss of the epoch (only available when mode is "eval")                * any keys written by 'on_epoch_end' of previous traces        