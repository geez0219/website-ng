# FastEstimator API reference

## Pipeline


#### class fastestimator.pipeline.pipeline.Pipeline(batch_size, feature_name, transform_train, transform_dataset=None, train_data=None, validation_data=None, data_filter=None, shuffle_buffer='auto', \*\*kwargs)
Bases: `object`

Class representing the data pipeline required for fastestimator


* **Parameters**

    * **batch_size** – Integer representing the batch size for training model

    * **feature_name** – List of strings representing the feature names in the data (headers in csv, keys in dictionary
      or features in TFRecords)

    * **transform_train** – List of lists of tensor transformations to be performed sequentially on the corresponding
      features.

    * **transform_dataset** – List of lists of numpy transformations to be performed sequentially  on the raw data
      before the TFRecords are made.

    * **train_data** – Training dataset in the form of dictionary containing numpy data, or csv file (with file
      paths or data)

    * **validation_data** – Validation data in the form of dictionary containing numpy data, or csv file, or fraction
      of training data to be sequestered for validation during training

    * **data_filter** – Filtering to be performed on the corresponding features in the form of an object from the Filter class

    * **shuffle_buffer** – buffer size for the shuffling, it can affect the memory consumption during training. default is ‘auto’.

    * **\*\*kwargs** – Additional arguments to be forwarded for the creation of TFRecords.



#### edit_feature(feature)
Can be overloaded to change raw data dictionary in any manner


* **Parameters**

    **feature** – Dictionary containing the raw data



* **Returns**

    Dictionary containing raw data to be stored in TFRecords



#### final_transform(preprocessed_data)
Can be overloaded to change tensors in any manner


* **Parameters**

    **preprocessed_data** – Batch of training data as a tf.data object



* **Returns**

    A dictionary of tensor data in the form of a tf.data object.



#### read_and_decode(dataset)
Reads and decodes the string data from TFRecords


* **Parameters**

    **dataset** – Dataset consisting of encoded data from TFRecords



* **Returns**

    Dictionary of decoded data



#### show_batches(mode='train', inputs=None, num_batches=1)
Shows batches of tensor data in numpy


* **Parameters**

    * **mode** – Mode for training (“train”, “eval” or “both”)

    * **inputs** – Directory for saving TFRecords

    * **num_batches** – Number of batches to show



* **Returns**

    A dictionary containing the batches numpy data with corresponding keys


## Dynamic Preprocess

### AbstractPreprocessing


#### class fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing()
Bases: `object`

An abstract class for preprocessing


#### transform(data, feature=None)
Placeholder function that is to be inherited by preprocessing classes.


* **Parameters**

    * **data** – Data to be preprocessed

    * **feature** – Auxiliary decoded data needed for the preprocessing



* **Returns**

    Transformed data numpy array


### NrrdReader


#### class fastestimator.pipeline.dynamic.preprocess.NrrdReader(parent_path='')
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`

Class for reading NRRD images


* **Parameters**

    **parent_path** (*str*) – Parent path that will be added on given path.



#### transform(path, feature=None)
Reads from NRRD image path


* **Parameters**

    * **path** – path of the NRRD image

    * **feature** – Auxiliary data that may be used by the image reader



* **Returns**

    Image as numpy array


### ImageReader


#### class fastestimator.pipeline.dynamic.preprocess.ImageReader(parent_path='', grey_scale=False)
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`

Class for reading png or jpg images


* **Parameters**

    * **parent_path** – Parent path that will be added on given path

    * **grey_scale** – Boolean to indicate whether or not to read image as grayscale



#### transform(path, feature=None)
Reads numpy array from image path


* **Parameters**

    * **path** – path of the image

    * **feature** – Auxiliary data that may be used by the image reader



* **Returns**

    Image as numpy array


### Zscore


#### class fastestimator.pipeline.dynamic.preprocess.Zscore()
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`

Standardize data using zscore method


#### transform(data, feature=None)
Standardizes the data


* **Parameters**

    * **data** – Data to be standardized

    * **feature** – Auxiliary data needed for the standardization



* **Returns**

    Array containing standardized data


### Minmax


#### class fastestimator.pipeline.dynamic.preprocess.Minmax()
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`

Normalize data using the minmax method


#### transform(data, feature=None)
Normalizes the data


* **Parameters**

    * **data** – Data to be normalized

    * **feature** – Auxiliary data needed for the normalization



* **Returns**

    Normalized numpy array


### Scale


#### class fastestimator.pipeline.dynamic.preprocess.Scale(scalar)
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`

Preprocessing class for scaling dataset


* **Parameters**

    **scalar** – Scalar for scaling the data



#### transform(data, feature=None)
Scales the data tensor


* **Parameters**

    * **data** – Data to be scaled

    * **feature** – Auxiliary data needed for the normalization



* **Returns**

    Scaled data array


### Onehot


#### class fastestimator.pipeline.dynamic.preprocess.Onehot(num_dim)
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`

Preprocessing class for converting categorical labels to onehot encoding


* **Parameters**

    **num_dim** – Number of dimensions of the labels



#### transform(data, feature=None)
Transforms categorical labels to onehot encodings


* **Parameters**

    * **data** – Data to be preprocessed

    * **feature** – Auxiliary data needed for the preprocessing



* **Returns**

    Transformed labels


### Resize


#### class fastestimator.pipeline.dynamic.preprocess.Resize(size, resize_method='bilinear')
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`


#### transform(data, feature=None)
Placeholder function that is to be inherited by preprocessing classes.


* **Parameters**

    * **data** – Data to be preprocessed

    * **feature** – Auxiliary decoded data needed for the preprocessing



* **Returns**

    Transformed data numpy array


### Reshape


#### class fastestimator.pipeline.dynamic.preprocess.Reshape(shape)
Bases: `fastestimator.pipeline.dynamic.preprocess.AbstractPreprocessing`

Preprocessing class for reshaping the data


* **Parameters**

    **shape** – target shape



#### transform(data, feature=None)
Reshapes data array


* **Parameters**

    * **data** – Data to be reshaped

    * **feature** – Auxiliary data needed for the resizing



* **Returns**

    Reshaped array


## Static Preprocess

### AbstractPreprocessing


#### class fastestimator.pipeline.static.preprocess.AbstractPreprocessing()
Bases: `object`

An abstract class for preprocessing


#### transform(data, decoded_data=None)
Placeholder function that is to be inherited by preprocessing classes.


* **Parameters**

    * **data** – Data to be preprocessed

    * **decoded_data** – Auxiliary decoded data needed for the preprocessing



* **Returns**

    Transformed data tensor


### Binarize


#### class fastestimator.pipeline.static.preprocess.Binarize(threshold)
Bases: `fastestimator.pipeline.static.preprocess.AbstractPreprocessing`

Binarize data based on threshold between 0 and 1


* **Parameters**

    **threshold** – Threshold for binarizing



#### transform(data, decoded_data=None)
Transforms the image to binary based on threshold


* **Parameters**

    * **data** – Data to be binarized

    * **decoded_data** – Auxiliary decoded data needed for the binarization



* **Returns**

    Tensor containing binarized data


### Zscore


#### class fastestimator.pipeline.static.preprocess.Zscore()
Bases: `fastestimator.pipeline.static.preprocess.AbstractPreprocessing`

Standardize data using zscore method


#### transform(data, decoded_data=None)
Standardizes the data tensor


* **Parameters**

    * **data** – Data to be standardized

    * **decoded_data** – Auxiliary decoded data needed for the standardization



* **Returns**

    Tensor containing standardized data


### Minmax


#### class fastestimator.pipeline.static.preprocess.Minmax()
Bases: `fastestimator.pipeline.static.preprocess.AbstractPreprocessing`

Normalize data using the minmax method


#### transform(data, decoded_data=None)
Normalizes the data tensor


* **Parameters**

    * **data** – Data to be normalized

    * **decoded_data** – Auxiliary decoded data needed for the normalization



* **Returns**

    Tensor after minmax


### Scale


#### class fastestimator.pipeline.static.preprocess.Scale(scalar)
Bases: `fastestimator.pipeline.static.preprocess.AbstractPreprocessing`

Preprocessing class for scaling dataset


* **Parameters**

    **scalar** – Scalar for scaling the data



#### transform(data, decoded_data=None)
Scales the data tensor


* **Parameters**

    * **data** – Data to be scaled

    * **decoded_data** – Auxiliary decoded data needed for the normalization



* **Returns**

    Scaled data tensor


### Onehot


#### class fastestimator.pipeline.static.preprocess.Onehot(num_dim)
Bases: `fastestimator.pipeline.static.preprocess.AbstractPreprocessing`

Preprocessing class for converting categorical labels to onehot encoding


* **Parameters**

    **num_dim** – Number of dimensions of the labels



#### transform(data, decoded_data=None)
Transforms categorical labels to onehot encodings


* **Parameters**

    * **data** – Data to be preprocessed

    * **decoded_data** – Auxiliary decoded data needed for the preprocessing



* **Returns**

    Transformed labels


### Resize


#### class fastestimator.pipeline.static.preprocess.Resize(size, resize_method=0)
Bases: `fastestimator.pipeline.static.preprocess.AbstractPreprocessing`

Preprocessing class for resizing the images


* **Parameters**

    * **size** – Destination shape of the images

    * **resize_method** – One of resize methods provided by tensorflow to be used



#### transform(data, decoded_data=None)
Resizes data tensor


* **Parameters**

    * **data** – Tensor to be resized

    * **decoded_data** – Auxiliary decoded data needed for the resizing



* **Returns**

    Resized tensor


### Reshape


#### class fastestimator.pipeline.static.preprocess.Reshape(shape)
Bases: `fastestimator.pipeline.static.preprocess.AbstractPreprocessing`

Preprocessing class for reshaping the data


* **Parameters**

    **shape** – target shape



#### transform(data, decoded_data=None)
Reshapes data tensor


* **Parameters**

    * **data** – Data to be reshaped

    * **decoded_data** – Auxiliary decoded data needed for the resizing



* **Returns**

    Reshaped tensor


## Augmentation

### AbstractAugmentation


#### class fastestimator.pipeline.static.augmentation.AbstractAugmentation(mode='train')
Bases: `object`

An abstract class for data augmentation that defines interfaces.
A custom augmentation can be defined by inheriting from this class.


* **Parameters**

    **mode** – Augmentation to be applied for training or evaluation, can be “train”, “eval” or “both”.



#### setup()
An interface method to be implemented by inheriting augmentation class to setup necessary parameters for the
augmentation


* **Returns**

    None



#### transform(data)
An interface method to be implemented by inheriting augmentation class to apply the transformation to data


* **Parameters**

    **data** – Data on which a transformation is to be applied



* **Returns**

    Transformed tensor


### Augmentation


#### class fastestimator.pipeline.static.augmentation.Augmentation(rotation_range=0.0, width_shift_range=0.0, height_shift_range=0.0, shear_range=0.0, zoom_range=1.0, flip_left_right=False, flip_up_down=False, mode='train')
Bases: `fastestimator.pipeline.static.augmentation.AbstractAugmentation`

This class supports commonly used 2D random affine transformations for data augmentation.
Either a scalar `x` or a tuple `[x1, x2]` can be specified for rotation, shearing, shifting, and zoom.


* **Parameters**

    * **rotation_range** – Scalar (x) that represents the range of random rotation (in degrees) from -x to x /
      Tuple ([x1, x2]) that represents  the range of random rotation between x1 and x2.

    * **width_shift_range** – Float (x) that represents the range of random width shift (in pixels) from -x to x /
      Tuple ([x1, x2]) that represents  the range of random width shift between x1 and x2.

    * **height_shift_range** – Float (x) that represents the range of random height shift (in pixels) from -x to x /
      Tuple ([x1, x2]) that represents  the range of random height shift between x1 and x2.

    * **shear_range** – Scalar (x) that represents the range of random shear (in degrees) from -x to x /
      Tuple ([x1, x2]) that represents  the range of random shear between x1 and x2.

    * **zoom_range** – Float (x) that represents the range of random zoom (in percentage) from -x to x /
      Tuple ([x1, x2]) that represents  the range of random zoom between x1 and x2.

    * **flip_left_right** – Boolean representing whether to flip the image horizontally with a probability of 0.5.

    * **flip_up_down** – Boolean representing whether to flip the image vertically with a probability of 0.5.

    * **mode** – Augmentation on ‘training’ data or ‘evaluation’ data.



#### flip()
Decides whether or not to flip


* **Returns**

    A boolean that represents whether or not to flip



#### rotate()
Creates affine transformation matrix for 2D rotation


* **Returns**

    Transform affine tensor



#### setup()
This method set the appropriate variables necessary for the random 2D augmentation. It also computes the
transformation matrix.


* **Returns**

    None



#### shear()
Creates affine transformation matrix for 2D shear


* **Returns**

    Transform affine tensor



#### shift()
Creates affine transformation matrix for 2D shift


* **Returns**

    Transform affine tensor



#### transform(data)
Transforms the data with the augmentation transformation


* **Parameters**

    **data** – Data to be transformed



* **Returns**

    Transformed (augmented) data



#### transform_matrix_offset_center(matrix)
Offsets the tensor to the center of the image


* **Parameters**

    **matrix** – Affine tensor



* **Returns**

    An affine tensor offset to the center of the image



#### zoom()
Creates affine transformation matrix for 2D zoom / scale


* **Returns**

    Transform affine tensor


## Filter


#### class fastestimator.pipeline.static.filter.Filter(feature_name, filter_value, keep_prob, mode='train')
Bases: `object`

Class for performing filtering on dataset based on scalar values.


* **Parameters**

    * **feature_name** – Name of the key in the dataset that is to be filtered

    * **filter_value** – The values in the dataset that are to be filtered.

    * **keep_prob** – The probability of keeping the example

    * **mode** – filter on ‘train’, ‘eval’ or ‘both’



#### predicate_fn(dataset)
Filters the dataset based on the filter probabilities.


* **Parameters**

    **dataset** – Tensorflow dataset object which is to be filtered



* **Returns**

    Tensorflow conditional for filtering the dataset based on the probabilities for each of the values.


## Cyclic Learning Rate


#### class fastestimator.network.lrscheduler.CyclicScheduler(num_cycle=1, cycle_multiplier=2, decrease_method='cosine')
Bases: `object`

A class representing cyclic learning rate scheduler


* **Parameters**

    * **num_cycle** – The number of cycles to be used by the learning rate scheduler

    * **cycle_multiplier** – The length of each next cycle’s multiplier

    * **decrease_method** – The decay method to be used with cyclic learning rate scheduler



#### lr_cosine_decay(global_steps, lr_ratio_start, lr_ratio_end, step_start, step_end)

#### lr_linear_decay(global_steps, lr_ratio_start, lr_ratio_end, step_start, step_end)

#### lr_schedule_fn(global_steps)
The actual function that computes the learning rate decay ratio using cyclic learning rate.


* **Parameters**

    **global_steps** – Current global step



* **Returns**

    Learning rate ratio


## Network


#### class fastestimator.network.network.Network(model, loss, metrics=None, loss_weights=None, optimizer='adam', model_dir=None)
Bases: `object`

Class for representing the model for fastestimator


* **Parameters**

    * **model** – An instance of tensorflow.keras model object.

    * **loss** – String or list or dictionary of strings representing a loss function (defined in keras)
      or it can be a function handle of a customized loss function that takes a true value and
      predicted value and returns a scalar loss value.

    * **metrics** – List or dictionary of strings representing metrics (defined in keras)
      or it can be a list of function handles of a customized metric function that
      takes a true values and predicted values and returns a scalar metric.

    * **loss_weights** – List of floats used only if the loss is a weighted loss
      with the individual components defined as a list in the “loss member variable” (default: `None`)

    * **optimizer** – the type the optimizer for the model in the form of a string.

    * **model_dir** – Directory where the model is to be saved (default is the temporary directory)


## Estimator


#### class fastestimator.estimator.estimator.Estimator(pipeline, network, epochs, steps_per_epoch=None, validation_steps=None, callbacks=[], log_steps=100)
Bases: `object`

`Estimator` class compiles all the components necessary to train a model.


* **Parameters**

    * **pipeline** – Object of the Pipeline class that consists of data parameters.

    * **network** – Object of the Network class that consists of the model definition and parameters.

    * **epochs** – Total number of training epochs

    * **steps_per_epoch** – The number batches in one epoch of training,
      if None, it will be automatically calculated. Evaluation is performed at the end of every epoch.
      (default: `None`)

    * **validation_steps** – Number of batches to be used for validation

    * **callbacks** – List of callbacks object in tf.keras. (default: `[]`)

    * **log_steps** – Number of steps after which training logs will be displayed periodically.



#### fit(inputs=None)
Function to perform training on the estimator


* **Parameters**

    **inputs** – Path to input data



* **Returns**

    None



#### train()
## Callbacks

### OutputLogger


#### class fastestimator.estimator.callbacks.OutputLogger(batch_size, log_steps=100, validation=True, num_process=1)
Bases: `tensorflow.python.keras.callbacks.Callback`

Keras callback for logging the output


* **Parameters**

    * **batch_size** – Size of the training batch

    * **log_steps** – Number of steps at which to output the logs

    * **validation** – Boolean representing whether or not to output validation information



#### on_batch_begin(batch, logs=None)
A backwards compatibility alias for on_train_batch_begin.


#### on_batch_end(batch, logs=None)
A backwards compatibility alias for on_train_batch_end.


#### on_epoch_end(epoch, logs=None)
Called at the end of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.


* **Parameters**

    * **epoch** – integer, index of epoch.

    * **logs** – dict, metric results for this training epoch, and for the
      validation epoch if validation is performed. Validation result keys
      are prefixed with val_.


### LearningRateUpdater


#### class fastestimator.estimator.callbacks.LearningRateUpdater(init_lr)
Bases: `tensorflow.python.keras.callbacks.Callback`

Keras callback to update the learning rate


* **Parameters**

    **init_lr** – initial learning rate



#### on_batch_begin(batch, logs=None)
A backwards compatibility alias for on_train_batch_begin.


#### on_batch_end(batch, logs={})
A backwards compatibility alias for on_train_batch_end.

### LearningRateScheduler


#### class fastestimator.estimator.callbacks.LearningRateScheduler(schedule)
Bases: `tensorflow.python.keras.callbacks.Callback`

Keras callback for the learning rate scheduler


* **Parameters**

    **schedule** – Schedule object to passed to the scheduler



#### on_batch_begin(batch, logs=None)
A backwards compatibility alias for on_train_batch_begin.


#### on_epoch_begin(epoch, logs=None)
Called at the start of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.


* **Parameters**

    * **epoch** – integer, index of epoch.

    * **logs** – dict. Currently no data is passed to this argument for this method
      but that may change in the future.


### ReduceLROnPlateau


#### class fastestimator.estimator.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, \*\*kwargs)
Bases: `tensorflow.python.keras.callbacks.Callback`

Keras callback for the reduce learning rate on pleateau


* **Parameters**

    * **monitor** – Metric to be monitored

    * **factor** – Factor by which to reduce learning rate

    * **patience** – Number of epochs to wait before reducing LR

    * **verbose** – Whether or not to output verbose logs

    * **mode** – Learning rate reduction mode

    * **min_delta** – Minimum significant difference

    * **cooldown** – 

    * **\*\*kwargs** – 



#### in_cooldown()

#### on_epoch_end(epoch, logs=None)
Called at the end of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.


* **Parameters**

    * **epoch** – integer, index of epoch.

    * **logs** – dict, metric results for this training epoch, and for the
      validation epoch if validation is performed. Validation result keys
      are prefixed with val_.



#### on_train_begin(logs=None)
Called at the beginning of training.

Subclasses should override for any actions to run.


* **Parameters**

    **logs** – dict. Currently no data is passed to this argument for this method
    but that may change in the future.


## TFRecord Utility Functions

### TFRecorder


#### class fastestimator.util.tfrecord.TFRecorder(train_data, feature_name, transform_dataset=None, validation_data=None, create_patch=False, max_tfrecord_mb=300, compression=None)
Bases: `object`

Class for creating TFRecords from numpy data or csv file containing paths to data on disk


* **Parameters**

    * **train_data** – Training dataset in the form of dictionary containing numpy data, or csv file (with file
      paths or data)

    * **feature_name** – List of strings representing the feature names in the data (headers in csv, keys in dictionary
      or features in TFRecords)

    * **transform_dataset** – List of lists of numpy transformations to be performed sequentially  on the raw data
      before the TFRecords are made.

    * **validation_data** – Validation data in the form of dictionary containing numpy data, or csv file, or fraction
      of training data to be sequestered for validation during training.

    * **create_patch** – Whether to create multiple records from single example.

    * **max_tfrecord_mb** – Maximum space to be occupied by one TFRecord.

    * **compression** – tfrecords compression type, one of None, ‘GZIP’ or ‘ZLIB’.



#### create_tfrecord(save_dir=None)

#### edit_feature(feature)
### tfrecord_to_np


#### class fastestimator.util.tfrecord.tfrecord_to_np()
Converts 1 TFRecord (created using fastestimator) to numpy data


* **Parameters**

    **file_path** – Path of TFRecord file



* **Returns**

    Dictionary containing numpy data


### get_number_of_examples


#### class fastestimator.util.tfrecord.get_number_of_examples()
Returns number of examples in 1 TFRecord


* **Parameters**

    * **file_path** – Path of TFRecord file

    * **show_warning** – Whether to display warning message

    * **compression** – None, ‘GZIP’ or ‘ZLIB’



* **Returns**

    Number of examples in the TFRecord


### get_features


#### class fastestimator.util.tfrecord.get_features()
Returns the feature information in TFRecords


* **Parameters**

    * **file_path** – Path of TFRecord file

    * **compression** – None, ‘GZIP’ or ‘ZLIB’



* **Returns**

    Dictionary containing feature information of TFRecords


### add_summary


#### class fastestimator.util.tfrecord.add_summary()
Adds summary.json file to existing path with tfrecords.


* **Parameters**

    * **data_dir** (*str*) – Folder path where tfrecords are stored.

    * **train_prefix** (*str*) – The prefix of all training tfrecord files.

    * **feature_name** (*list*) – Feature name in the tfrecord in a list.

    * **feature_dtype** (*list*) – Original data type for specific feature, this is used for decoding purpose.

    * **eval_prefix** (*str**, **optional*) – The prefix of all evaluation tfrecord files. Defaults to None.

    * **num_train_examples** (*int**, **optional*) – The total number of training examples, if None, it will calculate automatically. Defaults to None.

    * **num_eval_examples** (*int**, **optional*) – The total number of validation examples, if None, it will calculate automatically. Defaults to None.

    * **compression** (*str**, **optional*) – None, ‘GZIP’ or ‘ZLIB’. Defaults to None.
