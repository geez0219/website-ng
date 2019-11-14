# Image Segmation: Caltech-UCSD Birds 200 Dataset


```python
import os
import tempfile

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

import fastestimator as fe
from fastestimator.architecture.unet import UNet
from fastestimator.dataset import cub200
from fastestimator.trace import Dice, ModelSaver
from fastestimator.op.tensorop import BinaryCrossentropy, ModelOp, Minmax
from fastestimator.op.numpyop import ImageReader, MatReader, Reshape, Resize
from fastestimator.op import NumpyOp
from fastestimator.util import RecordWriter
```


```python
#parameters
epochs = 10
batch_size = 64
steps_per_epoch = None
validation_steps = None
```

## Download and prepare the CUB200 dataset

The cub200 dataset API will generate a summary CSV file for the data. The path of the csv file is returned as `csv_path`. The dataset path is returned as `path`. Inside the CSV file, the file paths are all relative to `path`. 


```python
csv_path, path = cub200.load_data()
```

## Create data pipeline

The `RecordWriter` will convert the data into TFRecord. You can specify your data preprocessing with the `Preprocess` ops before saving into TFRecord. 

Here the main task is to resize the images and masks into 128 by 128 pixels. The image and mask preprocessings are 

- **image**  
ImageReader &rarr; Resize

- **mask**    
MatReader &rarr; SelectDictKey (select only the mask info from MAT) &rarr; Resize &rarr; Reshape (add extra dimension)


We read the JPG images with `ImageReader`, the masks stored in MAT file with `MatReader`. There is other information stored in the MAT file, so we use the custom `SelectDictKey` op to retrieve the mask only.


```python
class SelectDictKey(NumpyOp):
    def forward(self, data, state):
        data = data['seg']
        return data
```

For each op, if the `inputs` argument is not provided, it defaults to output from the previous op. Similarly, if the `outputs` argument is not provided, it defaults to the input of current op. 


```python
writer = RecordWriter(
    save_dir=os.path.join(path, "tfrecords"),
    train_data=csv_path,
    validation_data=0.2,
    ops=[
        ImageReader(inputs='image', parent_path=path),
        Resize(target_size=(128, 128), keep_ratio=True, outputs='image'),
        MatReader(inputs='annotation', parent_path=path),
        SelectDictKey(),
        Resize((128, 128), keep_ratio=True),
        Reshape(shape=(128, 128, 1), outputs="annotation")
    ])
```

We can send this `RecordWriter` instance to `Pipeline`. `Pipeline` reads the TFRecord generated from `RecordWriter` and applies further transformation specified in the `ops` argument. In this case we read TFRecords and apply `Minmax` normalization to each image.


```python
pipeline = fe.Pipeline(batch_size=batch_size, data=writer, ops=Minmax(inputs='image', outputs='image'))
```

### Visualize data from pipeline

`Pipeline.show_results()` is useful for pipeline debugging. This method returns a list of batch data.


```python
sample_data = pipeline.show_results()
```

Let's convert the tensor into numpy array and plot the image and mask.


```python
img_batch = sample_data[0]['image'].numpy()
mask_batch = sample_data[0]['annotation'].numpy()
```


```python
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_batch[0])
axes[1].imshow(np.squeeze(mask_batch[0]), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f8f5415dba8>




![png](unet_cub200_files/unet_cub200_18_1.png)


## Create network

First, the `tf.keras.Model` and `tf.optimizers` are put together as `FEModel`.


```python
model = fe.build(model_def=UNet, model_name="unet_cub", optimizer=tf.optimizers.Adam(lr=0.001), loss_name="loss")
```

Then we combine the `FEModel` and `Loss` together as `Network`.


```python
network = fe.Network(ops=[
    ModelOp(inputs='image', model=model, outputs='mask_pred'),
    BinaryCrossentropy(y_true='annotation', y_pred='mask_pred', outputs="loss")
])
```

## Create Estimator

`Trace` is similar to callbacks in `tf.keras`. You can specify the metrics you would like to use. In this example we use Dice score. To save your model, you can use the `ModelSaver`. We will save our model in the `save_dir` folder.


```python
save_dir = tempfile.mkdtemp()
```


```python
traces = [
    Dice(true_key="annotation", pred_key='mask_pred'),
    ModelSaver(model_name="unet_cub", save_dir=save_dir, save_best='dice', save_best_mode='max')
]
```

After combining everything into `Estimator` we can start training!


```python
estimator = fe.Estimator(network=network, 
                         pipeline=pipeline, 
                         traces=traces, 
                         epochs=epochs, 
                         steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps)
```

## Start traininig


```python
estimator.fit()
```

## Inference


```python
model = load_model(os.path.join(save_dir, 'unet_cub_best_dice.h5'))
```


```python
predicted_mask = model.predict(img_batch)
```


```python
batch_index = 1
fig, axes = plt.subplots(1, 3)
axes[0].imshow(img_batch[batch_index])
axes[0].axis('off')
axes[0].set_title('original image', y=-0.3)
axes[1].imshow(np.squeeze(mask_batch[batch_index]), cmap='gray')
axes[1].axis('off')
axes[1].set_title('groundtruth', y=-0.3)
axes[2].imshow(np.squeeze(predicted_mask[batch_index]), cmap='gray')
axes[2].axis('off')
axes[2].set_title('segmentation mask', y=-0.3)
```




    Text(0.5, -0.3, 'segmentation mask')




![png](unet_cub200_files/unet_cub200_35_1.png)

