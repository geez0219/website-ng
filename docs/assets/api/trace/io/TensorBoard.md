## TensorBoard
```python
TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None)
```
Output data for use in TensorBoard.

#### Args:

* **log_dir (str, optional)** :  Path of the directory where to save the log files to be parsed by TensorBoard.        Defaults to 'logs'.
* **histogram_freq (int, optional)** :  Frequency (in epochs) at which to compute activation and weight histograms for        the layers of the model. If set to 0, histograms won't be computed. Defaults to 0.
* **write_graph (bool, optional)** :  Whether to visualize the graph in TensorBoard. The log file can become quite large        when write_graph is set to True. Defaults to True.
* **write_images (bool, str, list, optional)** :  If True will write model weights to visualize as an image in        TensorBoard. If a string or list of strings is provided, the corresponding keys will be written to        Tensorboard images. To get weights and specific keys use [True, 'key1', 'key2',...] Defaults to False.
* **update_freq (str, int, optional)** :  'batch' or 'epoch' or integer. When using 'batch', writes the losses and        metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 1000,        the callback will write the metrics and losses to TensorBoard every 1000 samples. Note that writing too        frequently to TensorBoard can slow down your training. Defaults to 'epoch'.
* **profile_batch (int, optional)** :  Which batch to run profiling on. 0 to disable. Note that FE batch numbering        starts from 0 rather than 1. Defaults to 2.
* **embeddings_freq (int, optional)** :  Frequency (in epochs) at which embedding layers will be visualized. If set to        0, embeddings won't be visualized.Defaults to 0.
* **embeddings_metadata (str, dict, optional)** :  A dictionary which maps layer name to a file name in which metadata        for this embedding layer is saved. See the details about metadata files format. In case if the same        metadata file is used for all embedding layers, string can be passed. Defaults to None.