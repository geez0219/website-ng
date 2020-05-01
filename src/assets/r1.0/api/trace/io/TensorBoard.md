## TensorBoard
```python
TensorBoard(log_dir:str='logs', update_freq:Union[NoneType, int, str]=100, write_graph:bool=True, write_images:Union[NoneType, str, List[str]]=None, weight_histogram_freq:Union[NoneType, int, str]=None, paint_weights:bool=False, write_embeddings:Union[NoneType, str, List[str]]=None, embedding_labels:Union[NoneType, str, List[str]]=None, embedding_images:Union[NoneType, str, List[str]]=None) -> None
```
Output data for use in TensorBoard.    Note that if you plan to run a tensorboard server simultaneous to training, you may want to consider using the
* **--reload_multifile=true flag until their multi-writer use case is finished** : 
* **https** : //github.com/tensorflow/tensorboard/issues/1063

#### Args:

* **log_dir** :  Path of the directory where the log files to be parsed by TensorBoard should be saved.
* **update_freq** :  'batch', 'epoch', integer, or strings like '10s', '15e'. When using 'batch', writes the losses and            metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000,            the callback will write the metrics and losses to TensorBoard every 1000 samples. You can also use strings            like '8s' to indicate every 8 steps or '5e' to indicate every 5 epochs. Note that writing too frequently to            TensorBoard can slow down your training. You can use None to disable updating, but this will make the trace            mostly useless.
* **write_graph** :  Whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph            is set to True.
* **write_images** :  If a string or list of strings is provided, the corresponding keys will be written to TensorBoard            images.
* **weight_histogram_freq** :  Frequency (in epochs) at which to compute activation and weight histograms for the layers            of the model. Same argument format as `update_freq`.
* **paint_weights** :  If True the system will attempt to visualize model weights as an image.
* **write_embeddings** :  If a string or list of strings is provided, the corresponding keys will be written to            TensorBoard embeddings.
* **embedding_labels** :  Keys corresponding to label information for the `write_embeddings`.
* **embedding_images** :  Keys corresponding to raw images to be associated with the `write_embeddings`.    