# Tutorial 10: Dataset with unpaired features

In almost all deep learning applications, different features within a dataset are paired together as single example. For instance, image and label(s) are paired in image classification, image and mask(s) are paired in image segmentation.

However, in image-image translation, sometimes the features in dataset are unpaired. For example, we may have 500 horse images for 700 zebra images. During the training, we need to randomly select one horse image and one zebra image. 

In FastEstimator, unpaired features are handled by `RecordWriter`. If there are multiple unpaired features, express them as __tuple__  in the `train_data`, `validation_data` and `ops` argument of `RecordWriter`.


```python
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import fastestimator as fe
```

## Step 0 - Data preparation and visualization
We will here use the horse2zebra dataset.


```python
from fastestimator.dataset.horse2zebra import load_data

# Use load_data from our dataset API to load the dataset.
trainA_csv, trainB_csv, valA_csv, valB_csv, parent_path = load_data()
print("horse image is stored in {}".format(trainA_csv))
print("zebra image is stored in {}".format(trainB_csv))
```

    horse image is stored in /home/ubuntu/fastestimator_data/horse2zebra/trainA.csv
    zebra image is stored in /home/ubuntu/fastestimator_data/horse2zebra/trainB.csv



```python
# Let's take a look at the data, by loading the csv file with all images path information.
df_train_A = pd.read_csv(trainA_csv)
df_train_B = pd.read_csv(trainB_csv)

fig, axes = plt.subplots(1, 2)
axes[0].axis('off')
axes[1].axis('off')

# We select one image of horse and one of zebra and plot them.
img1 = plt.imread(os.path.join(parent_path, df_train_A["imgA"][2]))
axes[0].imshow(img1)

img2 = plt.imread(os.path.join(parent_path, df_train_B["imgB"][2]))
axes[1].imshow(img2)
```




    <matplotlib.image.AxesImage at 0x7f50b5ca1b70>




![png](assets/tutorial/t10_unpaired_dataset_files/t10_unpaired_dataset_4_1.png)


## Step 1 - RecordWriter: read unpaired features using a tuple

We deal with the unpaired images in RecordWriter. In ops, we specify a tuple of two ops: to load the first image and to load the second one (here using ImageReader).


```python
from fastestimator.op.numpyop import ImageReader
from fastestimator import RecordWriter

target_dir = os.path.join(parent_path, 'FEdata')

# Check whether the target folder already exists. RecordWriter either needs an empty or non-exist target folder.    
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# Create a RecordWriter with a tuple of two ops to pair images.
writer = RecordWriter(save_dir=target_dir,
                         train_data=(trainA_csv, trainB_csv),   #this is a tuple
                         validation_data = (valA_csv, valB_csv), #this is a tuple
                         ops=([ImageReader(inputs="imgA", outputs="imgA", parent_path=parent_path)], # first tuple element
                              [ImageReader(inputs="imgB", outputs="imgB", parent_path=parent_path)])) # second tuple element
```


```python
# We write the data to the disk using the write method
writer.write()
```

    FastEstimator: Saving tfrecord to /home/ubuntu/fastestimator_data/horse2zebra/FEdata
    FastEstimator: Converting Train TFRecords 0.0%, Speed: 0.00 record/sec
    FastEstimator: Converting Train TFRecords 4.9%, Speed: 1007.86 record/sec
    FastEstimator: Converting Train TFRecords 9.7%, Speed: 1078.52 record/sec
    FastEstimator: Converting Train TFRecords 14.6%, Speed: 1134.03 record/sec
    FastEstimator: Converting Train TFRecords 19.5%, Speed: 1159.41 record/sec
    FastEstimator: Converting Train TFRecords 24.3%, Speed: 1164.55 record/sec
    FastEstimator: Converting Train TFRecords 29.2%, Speed: 1179.34 record/sec
    FastEstimator: Converting Train TFRecords 34.1%, Speed: 1174.25 record/sec
    FastEstimator: Converting Train TFRecords 39.0%, Speed: 1161.07 record/sec
    FastEstimator: Converting Train TFRecords 43.8%, Speed: 1150.87 record/sec
    FastEstimator: Converting Train TFRecords 48.7%, Speed: 1146.21 record/sec
    FastEstimator: Converting Train TFRecords 53.6%, Speed: 1143.16 record/sec
    FastEstimator: Converting Train TFRecords 58.4%, Speed: 1140.97 record/sec
    FastEstimator: Converting Train TFRecords 63.3%, Speed: 1134.64 record/sec
    FastEstimator: Converting Train TFRecords 68.2%, Speed: 1132.32 record/sec
    FastEstimator: Converting Train TFRecords 73.0%, Speed: 1133.52 record/sec
    FastEstimator: Converting Train TFRecords 77.9%, Speed: 1128.43 record/sec
    FastEstimator: Converting Train TFRecords 82.8%, Speed: 1129.61 record/sec
    FastEstimator: Converting Train TFRecords 87.6%, Speed: 1132.10 record/sec
    FastEstimator: Converting Train TFRecords 92.5%, Speed: 1141.80 record/sec
    FastEstimator: Converting Train TFRecords 97.4%, Speed: 1167.16 record/sec
    FastEstimator: Converting Eval TFRecords 0.0%, Speed: 0.00 record/sec
    FastEstimator: Converting Eval TFRecords 3.3%, Speed: 345.71 record/sec
    FastEstimator: Converting Eval TFRecords 6.7%, Speed: 425.85 record/sec
    FastEstimator: Converting Eval TFRecords 10.0%, Speed: 508.26 record/sec
    FastEstimator: Converting Eval TFRecords 13.3%, Speed: 560.97 record/sec
    FastEstimator: Converting Eval TFRecords 16.7%, Speed: 589.10 record/sec
    FastEstimator: Converting Eval TFRecords 20.0%, Speed: 608.73 record/sec
    FastEstimator: Converting Eval TFRecords 23.3%, Speed: 631.43 record/sec
    FastEstimator: Converting Eval TFRecords 26.7%, Speed: 635.69 record/sec
    FastEstimator: Converting Eval TFRecords 30.0%, Speed: 650.80 record/sec
    FastEstimator: Converting Eval TFRecords 33.3%, Speed: 655.65 record/sec
    FastEstimator: Converting Eval TFRecords 36.7%, Speed: 666.35 record/sec
    FastEstimator: Converting Eval TFRecords 40.0%, Speed: 674.80 record/sec
    FastEstimator: Converting Eval TFRecords 43.3%, Speed: 682.64 record/sec
    FastEstimator: Converting Eval TFRecords 46.7%, Speed: 696.64 record/sec
    FastEstimator: Converting Eval TFRecords 50.0%, Speed: 699.48 record/sec
    FastEstimator: Converting Eval TFRecords 53.3%, Speed: 709.59 record/sec
    FastEstimator: Converting Eval TFRecords 56.7%, Speed: 717.45 record/sec
    FastEstimator: Converting Eval TFRecords 60.0%, Speed: 702.22 record/sec
    FastEstimator: Converting Eval TFRecords 63.3%, Speed: 716.01 record/sec
    FastEstimator: Converting Eval TFRecords 66.7%, Speed: 732.11 record/sec
    FastEstimator: Converting Eval TFRecords 70.0%, Speed: 743.40 record/sec
    FastEstimator: Converting Eval TFRecords 73.3%, Speed: 758.51 record/sec
    FastEstimator: Converting Eval TFRecords 76.7%, Speed: 774.56 record/sec
    FastEstimator: Converting Eval TFRecords 80.0%, Speed: 785.45 record/sec
    FastEstimator: Converting Eval TFRecords 83.3%, Speed: 795.00 record/sec
    FastEstimator: Converting Eval TFRecords 86.7%, Speed: 805.67 record/sec
    FastEstimator: Converting Eval TFRecords 90.0%, Speed: 823.15 record/sec
    FastEstimator: Converting Eval TFRecords 93.3%, Speed: 834.20 record/sec
    FastEstimator: Converting Eval TFRecords 96.7%, Speed: 848.07 record/sec
    FastEstimator: Converting Train TFRecords 0.0%, Speed: 0.00 record/sec
    FastEstimator: Converting Train TFRecords 4.8%, Speed: 1025.53 record/sec
    FastEstimator: Converting Train TFRecords 9.6%, Speed: 1060.27 record/sec
    FastEstimator: Converting Train TFRecords 14.4%, Speed: 1089.73 record/sec
    FastEstimator: Converting Train TFRecords 19.2%, Speed: 1107.58 record/sec
    FastEstimator: Converting Train TFRecords 24.0%, Speed: 1113.61 record/sec
    FastEstimator: Converting Train TFRecords 28.7%, Speed: 1112.07 record/sec
    FastEstimator: Converting Train TFRecords 33.5%, Speed: 1115.08 record/sec
    FastEstimator: Converting Train TFRecords 38.3%, Speed: 1127.91 record/sec
    FastEstimator: Converting Train TFRecords 43.1%, Speed: 1127.72 record/sec
    FastEstimator: Converting Train TFRecords 47.9%, Speed: 1130.52 record/sec
    FastEstimator: Converting Train TFRecords 52.7%, Speed: 1137.09 record/sec
    FastEstimator: Converting Train TFRecords 57.5%, Speed: 1134.69 record/sec
    FastEstimator: Converting Train TFRecords 62.3%, Speed: 1133.06 record/sec
    FastEstimator: Converting Train TFRecords 67.1%, Speed: 1137.70 record/sec
    FastEstimator: Converting Train TFRecords 71.9%, Speed: 1140.89 record/sec
    FastEstimator: Converting Train TFRecords 76.6%, Speed: 1142.70 record/sec
    FastEstimator: Converting Train TFRecords 81.4%, Speed: 1147.54 record/sec
    FastEstimator: Converting Train TFRecords 86.2%, Speed: 1149.94 record/sec
    FastEstimator: Converting Train TFRecords 91.0%, Speed: 1165.62 record/sec
    FastEstimator: Converting Train TFRecords 95.8%, Speed: 1189.71 record/sec
    FastEstimator: Converting Eval TFRecords 0.0%, Speed: 0.00 record/sec
    FastEstimator: Converting Eval TFRecords 2.9%, Speed: 490.30 record/sec
    FastEstimator: Converting Eval TFRecords 5.7%, Speed: 596.38 record/sec
    FastEstimator: Converting Eval TFRecords 8.6%, Speed: 596.32 record/sec
    FastEstimator: Converting Eval TFRecords 11.4%, Speed: 633.85 record/sec
    FastEstimator: Converting Eval TFRecords 14.3%, Speed: 666.14 record/sec
    FastEstimator: Converting Eval TFRecords 17.1%, Speed: 649.50 record/sec
    FastEstimator: Converting Eval TFRecords 20.0%, Speed: 684.49 record/sec
    FastEstimator: Converting Eval TFRecords 22.9%, Speed: 663.74 record/sec
    FastEstimator: Converting Eval TFRecords 25.7%, Speed: 677.56 record/sec
    FastEstimator: Converting Eval TFRecords 28.6%, Speed: 678.18 record/sec
    FastEstimator: Converting Eval TFRecords 31.4%, Speed: 666.21 record/sec
    FastEstimator: Converting Eval TFRecords 34.3%, Speed: 678.66 record/sec
    FastEstimator: Converting Eval TFRecords 37.1%, Speed: 669.46 record/sec
    FastEstimator: Converting Eval TFRecords 40.0%, Speed: 665.62 record/sec
    FastEstimator: Converting Eval TFRecords 42.9%, Speed: 642.09 record/sec
    FastEstimator: Converting Eval TFRecords 45.7%, Speed: 638.64 record/sec
    FastEstimator: Converting Eval TFRecords 48.6%, Speed: 633.91 record/sec
    FastEstimator: Converting Eval TFRecords 51.4%, Speed: 641.46 record/sec
    FastEstimator: Converting Eval TFRecords 54.3%, Speed: 647.44 record/sec
    FastEstimator: Converting Eval TFRecords 57.1%, Speed: 653.32 record/sec
    FastEstimator: Converting Eval TFRecords 60.0%, Speed: 661.63 record/sec
    FastEstimator: Converting Eval TFRecords 62.9%, Speed: 668.67 record/sec
    FastEstimator: Converting Eval TFRecords 65.7%, Speed: 675.55 record/sec
    FastEstimator: Converting Eval TFRecords 68.6%, Speed: 685.16 record/sec
    FastEstimator: Converting Eval TFRecords 71.4%, Speed: 692.89 record/sec
    FastEstimator: Converting Eval TFRecords 74.3%, Speed: 707.21 record/sec
    FastEstimator: Converting Eval TFRecords 77.1%, Speed: 719.38 record/sec
    FastEstimator: Converting Eval TFRecords 80.0%, Speed: 728.38 record/sec
    FastEstimator: Converting Eval TFRecords 82.9%, Speed: 741.51 record/sec
    FastEstimator: Converting Eval TFRecords 85.7%, Speed: 755.31 record/sec
    FastEstimator: Converting Eval TFRecords 88.6%, Speed: 768.73 record/sec
    FastEstimator: Converting Eval TFRecords 91.4%, Speed: 778.49 record/sec
    FastEstimator: Converting Eval TFRecords 94.3%, Speed: 789.66 record/sec
    FastEstimator: Converting Eval TFRecords 97.1%, Speed: 799.88 record/sec



```python

```
