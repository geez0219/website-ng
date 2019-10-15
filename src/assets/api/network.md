build
```python
build(model_def, model_name, optimizer, loss_name)
build keras model instance in FastEstimator

Args:
    model_def (function): function definition of tf.keras model
    model_name (str, list, tuple): model name(s)
    optimizer (str, optimizer, list, tuple): optimizer(s)
    loss_name (str, list, tuple): loss name(s)

Returns:
    model: model(s) compiled by FastEstimator***
```
Network
```python
Network(ops)
A class representing network operations for FastEstimator model training.
```
Args:
    ops : Specifies the series of operations for training modelload_epoch
```python
load_epoch(self, epoch, mode)
This function loads stable computational graph for the current epoch.

Args:
    epoch: Training epoch number
    mode: 'train' or 'eval'

Returns:
     list of the models, epoch lossesprepare
```python
prepare(self)
This function constructs the model specified in model definition and create replica of model
for distributed training across multiple devices if there are multiple GPU available.run_step
```python
run_step(self, batch, ops, model_list, epoch_losses, state, warm_up=False)
Function that calculates the loss and gradients for curent step in training. It also constructs the higher
level computational graph between the models before the training.

Args:
    batch : dictionary that contains batch data and predictions from last epoch
    ops : Model operation dictionary that contains 'Inputs','Mode', and 'Outputs'
    model_list : List of the models
    epoch_losses : List of epoch losses.
    state : run time dictionary that contains following keys 'mode' and 'batch size'
    warm_up (bool, optional): Specifies if it's in warm up phase or not. Defaults to False.

Returns:
    dictionary containing the predictions of current epoch