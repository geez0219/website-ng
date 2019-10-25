## Network
```python
Network(ops)
```
A class representing network operations for FastEstimator model training.

#### Args:

* **ops** :  Specifies the series of operations for training model

### load_epoch
```python
load_epoch(self, epoch, mode)
```
 This function loads stable computational graph for the current epoch.

#### Args:

* **epoch** :  Training epoch number
* **mode** :  'train' or 'eval'

#### Returns:
             list of the models, epoch losses        

### prepare
```python
prepare(self, mode_list)
```
This function constructs the operations necessary for each epoch        

### run_step
```python
run_step(self, batch, ops, model_list, epoch_losses, state)
```
Function that calculates the loss and gradients for curent step in training. It also constructs the higher        level computational graph between the models before the training.

#### Args:

* **batch** :  dictionary that contains batch data and predictions from last epoch
* **ops** :  Model operation dictionary that contains 'Inputs','Mode', and 'Outputs'
* **model_list** :  List of the models
* **epoch_losses** :  List of epoch losses.
* **state** :  run time dictionary that contains following keys 'mode' and 'batch size'

#### Returns:
            dictionary containing the predictions of current epoch        