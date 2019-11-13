## CyclicLRSchedule
```python
CyclicLRSchedule(num_cycle=1, cycle_multiplier=2, decrease_method='cosine')
```
A class representing cyclic learning rate scheduler

#### Args:

* **num_cycle** :  The number of cycles to be used by the learning rate scheduler
* **cycle_multiplier** :  Multiplier of the next cycle length with respect to previous cycle length
* **decrease_method** :  The decay method to be used with cyclic learning rate scheduler

### lr_cosine_decay
```python
lr_cosine_decay(self, current_step, lr_start, lr_end, step_start, step_end)
```
This function applies cosine decay to the learning rate

#### Args:

* **current_step** :  Current step of the training epoch
* **lr_start** :  Learning rate from where it will start decaying
* **lr_end** :  Learning rate till which it will decay
* **step_start** :  Beginning step in the cycle of the learning rate scheduler
* **step_end** :  Last step in the cycle of the learning rate schedular

#### Returns:
            Decayed learning rate        

### lr_linear_decay
```python
lr_linear_decay(self, current_step, lr_start, lr_end, step_start, step_end)
```
This function applies linear decay to the learning rate

#### Args:

* **current_step** :  Current step of the training epoch
* **lr_start** :  Learning rate from where it will start decaying
* **lr_end** :  Learning rate till which it will decay
* **step_start** :  Beginning step in the cycle of the learning rate scheduler
* **step_end** :  Last step in the cycle of the learning rate schedular

#### Returns:
            Decayed learning rate        

### schedule_fn
```python
schedule_fn(self, current_step_or_epoch, lr)
```
The function computes the learning rate decay ratio using cyclic learning rate

#### Args:

* **current_step_or_epoch** :  Current training step or epoch
* **lr** :  Current learning rate

#### Returns:
            Learning rate ratio        