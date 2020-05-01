<h1>Lung segmentation using montgomery dataset</h1>


```python
import os
import tempfile
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

import fastestimator as fe
from fastestimator.architecture.pytorch import UNet
from fastestimator.dataset.data import montgomery
from fastestimator.op.numpyop import Delete, NumpyOp
from fastestimator.op.numpyop.meta import Sometimes
from fastestimator.op.numpyop.multivariate import HorizontalFlip, Resize, Rotate
from fastestimator.op.numpyop.univariate import Minmax, ReadImage, Reshape
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Dice
```


```python
pd.set_option('display.max_colwidth', 500)
```


```python
batch_size = 4
epochs = 25
max_steps_per_epoch = None
save_dir = tempfile.mkdtemp()
data_dir = None
```

# Download Data

We download the Montgomery data first:


```python
csv = montgomery.load_data(root_dir=data_dir)
```

This creates a `CSVDataset`, let's see what is inside:


```python
df = pd.DataFrame.from_dict(csv.data, orient='index')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>mask_left</th>
      <th>mask_right</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MontgomerySet/CXR_png/MCUCXR_0243_1.png</td>
      <td>MontgomerySet/ManualMask/leftMask/MCUCXR_0243_1.png</td>
      <td>MontgomerySet/ManualMask/rightMask/MCUCXR_0243_1.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MontgomerySet/CXR_png/MCUCXR_0022_0.png</td>
      <td>MontgomerySet/ManualMask/leftMask/MCUCXR_0022_0.png</td>
      <td>MontgomerySet/ManualMask/rightMask/MCUCXR_0022_0.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MontgomerySet/CXR_png/MCUCXR_0086_0.png</td>
      <td>MontgomerySet/ManualMask/leftMask/MCUCXR_0086_0.png</td>
      <td>MontgomerySet/ManualMask/rightMask/MCUCXR_0086_0.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MontgomerySet/CXR_png/MCUCXR_0008_0.png</td>
      <td>MontgomerySet/ManualMask/leftMask/MCUCXR_0008_0.png</td>
      <td>MontgomerySet/ManualMask/rightMask/MCUCXR_0008_0.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MontgomerySet/CXR_png/MCUCXR_0094_0.png</td>
      <td>MontgomerySet/ManualMask/leftMask/MCUCXR_0094_0.png</td>
      <td>MontgomerySet/ManualMask/rightMask/MCUCXR_0094_0.png</td>
    </tr>
  </tbody>
</table>
</div>



# Building Components

We are going to setup the stage for training. 

## Step 1: Create Pipeline


```python
class CombineLeftRightMask(NumpyOp):    
    def forward(self, data: List[np.ndarray], state: Dict[str, Any]) -> List[np.ndarray]:
        mask_left, mask_right = data
        data = mask_left + mask_right
        return data
```


```python
pipeline = fe.Pipeline(
    train_data=csv,
    eval_data=csv.split(0.2),
    batch_size=batch_size,
    ops=[
        ReadImage(inputs="image", parent_path=csv.parent_path, outputs="image", grey_scale=True),
        ReadImage(inputs="mask_left", parent_path=csv.parent_path, outputs="mask_left", grey_scale=True, mode='!infer'),
        ReadImage(inputs="mask_right",
                  parent_path=csv.parent_path,
                  outputs="mask_right",
                  grey_scale=True,
                  mode='!infer'),
        CombineLeftRightMask(inputs=("mask_left", "mask_right"), outputs="mask", mode='!infer'),
        Delete(keys=["mask_left", "mask_right"], mode='!infer'),
        Resize(image_in="image", width=512, height=512),
        Resize(image_in="mask", width=512, height=512, mode='!infer'),
        Sometimes(numpy_op=HorizontalFlip(image_in="image", mask_in="mask", mode='train')),
        Sometimes(numpy_op=Rotate(
            image_in="image", mask_in="mask", limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, mode='train')),
        Minmax(inputs="image", outputs="image"),
        Minmax(inputs="mask", outputs="mask", mode='!infer'),
        Reshape(shape=(1, 512, 512), inputs="image", outputs="image"),
        Reshape(shape=(1, 512, 512), inputs="mask", outputs="mask", mode='!infer')
    ])
```

Let's see if the `Pipeline` output is reasonable. We call `get_results` to get outputs from `Pipeline`.


```python
batch_data = pipeline.get_results()
```


```python
batch_index = 1
fig, ax = plt.subplots(1, 2, figsize=(15,6))
ax[0].imshow(np.squeeze(batch_data['image'][batch_index]), cmap='gray')
ax[1].imshow(np.squeeze(batch_data['mask'][batch_index]), cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f62440ace80>




![png](assets/example/semantic_segmentation/unet_files/unet_17_1.png)


## Step 2: Create Network


```python
model = fe.build(
    model_fn=lambda: UNet(input_size=(1, 512, 512)),
    optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=0.0001),
    model_names="lung_segmentation"
)
```


```python
network = fe.Network(ops=[
    ModelOp(inputs="image", model=model, outputs="pred_segment"),
    CrossEntropy(inputs=("pred_segment", "mask"), outputs="loss", form="binary"),
    UpdateOp(model=model, loss_name="loss")
])
```

## Step 3: Create Estimator


```python
traces = [
    Dice(true_key="mask", pred_key="pred_segment"),
    BestModelSaver(model=model, save_dir=save_dir, metric='Dice', save_best_mode='max')
]
```


```python
estimator = fe.Estimator(network=network,
                         pipeline=pipeline,
                         epochs=epochs,
                         log_steps=20,
                         traces=traces,
                         max_steps_per_epoch=max_steps_per_epoch)
```

# Training


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    FastEstimator-Start: step: 0; lung_segmentation_lr: 0.0001; 
    FastEstimator-Train: step: 0; loss: 0.65074736; 
    FastEstimator-Train: step: 20; loss: 0.32184097; steps/sec: 2.84; 
    FastEstimator-Train: step: 28; epoch: 0; epoch_time: 12.08 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 28; epoch: 0; loss: 0.6737924; min_loss: 0.6737924; since_best: 0; dice: 1.7335249448973935e-13; 
    FastEstimator-Train: step: 40; loss: 0.3740636; steps/sec: 2.14; 
    FastEstimator-Train: step: 56; epoch: 1; epoch_time: 12.1 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 56; epoch: 1; loss: 0.32339695; min_loss: 0.32339695; since_best: 0; dice: 0.07768369491807968; 
    FastEstimator-Train: step: 60; loss: 0.44676578; steps/sec: 2.16; 
    FastEstimator-Train: step: 80; loss: 0.14473557; steps/sec: 2.83; 
    FastEstimator-Train: step: 84; epoch: 2; epoch_time: 12.12 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 84; epoch: 2; loss: 0.12985013; min_loss: 0.12985013; since_best: 0; dice: 0.8881443049809153; 
    FastEstimator-Train: step: 100; loss: 0.1680123; steps/sec: 2.12; 
    FastEstimator-Train: step: 112; epoch: 3; epoch_time: 12.24 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 112; epoch: 3; loss: 0.12533109; min_loss: 0.12533109; since_best: 0; dice: 0.8912838752069556; 
    FastEstimator-Train: step: 120; loss: 0.09625991; steps/sec: 2.11; 
    FastEstimator-Train: step: 140; epoch: 4; epoch_time: 12.45 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 140; epoch: 4; loss: 0.1327874; min_loss: 0.12533109; since_best: 1; dice: 0.9037457108044139; 
    FastEstimator-Train: step: 140; loss: 0.100700244; steps/sec: 2.11; 
    FastEstimator-Train: step: 160; loss: 0.076236516; steps/sec: 2.78; 
    FastEstimator-Train: step: 168; epoch: 5; epoch_time: 12.42 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 168; epoch: 5; loss: 0.12693708; min_loss: 0.12533109; since_best: 2; dice: 0.9092143670270832; 
    FastEstimator-Train: step: 180; loss: 0.074325085; steps/sec: 2.11; 
    FastEstimator-Train: step: 196; epoch: 6; epoch_time: 12.26 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 196; epoch: 6; loss: 0.08061478; min_loss: 0.08061478; since_best: 0; dice: 0.9364716513786071; 
    FastEstimator-Train: step: 200; loss: 0.050859306; steps/sec: 2.04; 
    FastEstimator-Train: step: 220; loss: 0.0795434; steps/sec: 2.81; 
    FastEstimator-Train: step: 224; epoch: 7; epoch_time: 12.73 sec; 
    FastEstimator-Eval: step: 224; epoch: 7; loss: 0.077932164; min_loss: 0.077932164; since_best: 0; dice: 0.9363544687013216; 
    FastEstimator-Train: step: 240; loss: 0.04766827; steps/sec: 2.09; 
    FastEstimator-Train: step: 252; epoch: 8; epoch_time: 12.39 sec; 
    FastEstimator-Eval: step: 252; epoch: 8; loss: 0.08354305; min_loss: 0.077932164; since_best: 1; dice: 0.9357203667668574; 
    FastEstimator-Train: step: 260; loss: 0.07521939; steps/sec: 1.98; 
    FastEstimator-Train: step: 280; epoch: 9; epoch_time: 12.96 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 280; epoch: 9; loss: 0.07668398; min_loss: 0.07668398; since_best: 0; dice: 0.9423949873504999; 
    FastEstimator-Train: step: 280; loss: 0.040693548; steps/sec: 2.13; 
    FastEstimator-Train: step: 300; loss: 0.056012712; steps/sec: 2.8; 
    FastEstimator-Train: step: 308; epoch: 10; epoch_time: 12.25 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 308; epoch: 10; loss: 0.07509731; min_loss: 0.07509731; since_best: 0; dice: 0.9461569423558578; 
    FastEstimator-Train: step: 320; loss: 0.04584109; steps/sec: 1.99; 
    FastEstimator-Train: step: 336; epoch: 11; epoch_time: 12.89 sec; 
    FastEstimator-Eval: step: 336; epoch: 11; loss: 0.071265906; min_loss: 0.071265906; since_best: 0; dice: 0.9425017790028412; 
    FastEstimator-Train: step: 340; loss: 0.06591544; steps/sec: 2.06; 
    FastEstimator-Train: step: 360; loss: 0.039461873; steps/sec: 2.77; 
    FastEstimator-Train: step: 364; epoch: 12; epoch_time: 12.69 sec; 
    FastEstimator-Eval: step: 364; epoch: 12; loss: 0.06946295; min_loss: 0.06946295; since_best: 0; dice: 0.9453714568046383; 
    FastEstimator-Train: step: 380; loss: 0.053401247; steps/sec: 2.03; 
    FastEstimator-Train: step: 392; epoch: 13; epoch_time: 12.69 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 392; epoch: 13; loss: 0.06035184; min_loss: 0.06035184; since_best: 0; dice: 0.9530510743823034; 
    FastEstimator-Train: step: 400; loss: 0.032455094; steps/sec: 2.12; 
    FastEstimator-Train: step: 420; epoch: 14; epoch_time: 12.33 sec; 
    FastEstimator-Eval: step: 420; epoch: 14; loss: 0.072278894; min_loss: 0.06035184; since_best: 1; dice: 0.9376849732175224; 
    FastEstimator-Train: step: 420; loss: 0.053798117; steps/sec: 2.07; 
    FastEstimator-Train: step: 440; loss: 0.038134858; steps/sec: 2.77; 
    FastEstimator-Train: step: 448; epoch: 15; epoch_time: 12.62 sec; 
    FastEstimator-Eval: step: 448; epoch: 15; loss: 0.06692966; min_loss: 0.06035184; since_best: 2; dice: 0.9515013302881467; 
    FastEstimator-Train: step: 460; loss: 0.039841093; steps/sec: 2.09; 
    FastEstimator-Train: step: 476; epoch: 16; epoch_time: 12.42 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 476; epoch: 16; loss: 0.062103588; min_loss: 0.06035184; since_best: 3; dice: 0.9542675866982232; 
    FastEstimator-Train: step: 480; loss: 0.029947285; steps/sec: 2.14; 
    FastEstimator-Train: step: 500; loss: 0.027964272; steps/sec: 2.78; 
    FastEstimator-Train: step: 504; epoch: 17; epoch_time: 12.24 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 504; epoch: 17; loss: 0.05789433; min_loss: 0.05789433; since_best: 0; dice: 0.9561937779850502; 
    FastEstimator-Train: step: 520; loss: 0.030686911; steps/sec: 2.12; 
    FastEstimator-Train: step: 532; epoch: 18; epoch_time: 12.29 sec; 
    FastEstimator-Eval: step: 532; epoch: 18; loss: 0.05837614; min_loss: 0.05789433; since_best: 1; dice: 0.9551540080564349; 
    FastEstimator-Train: step: 540; loss: 0.02688715; steps/sec: 2.12; 
    FastEstimator-Train: step: 560; epoch: 19; epoch_time: 12.26 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 560; epoch: 19; loss: 0.05515228; min_loss: 0.05515228; since_best: 0; dice: 0.9569575317963865; 
    FastEstimator-Train: step: 560; loss: 0.045194946; steps/sec: 2.12; 
    FastEstimator-Train: step: 580; loss: 0.029270299; steps/sec: 2.81; 
    FastEstimator-Train: step: 588; epoch: 20; epoch_time: 12.29 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 588; epoch: 20; loss: 0.054214593; min_loss: 0.054214593; since_best: 0; dice: 0.9588091257130298; 
    FastEstimator-Train: step: 600; loss: 0.03187306; steps/sec: 2.15; 
    FastEstimator-Train: step: 616; epoch: 21; epoch_time: 12.14 sec; 
    FastEstimator-Eval: step: 616; epoch: 21; loss: 0.056209736; min_loss: 0.054214593; since_best: 1; dice: 0.9565499194603396; 
    FastEstimator-Train: step: 620; loss: 0.03272804; steps/sec: 2.16; 
    FastEstimator-Train: step: 640; loss: 0.034923207; steps/sec: 2.82; 
    FastEstimator-Train: step: 644; epoch: 22; epoch_time: 12.12 sec; 
    FastEstimator-Eval: step: 644; epoch: 22; loss: 0.05843257; min_loss: 0.054214593; since_best: 2; dice: 0.9549386241705727; 
    FastEstimator-Train: step: 660; loss: 0.03908976; steps/sec: 2.14; 
    FastEstimator-Train: step: 672; epoch: 23; epoch_time: 12.16 sec; 
    Saved model to /tmp/tmpnogbnnb5/lung_segmentation_best_dice.pt
    FastEstimator-Eval: step: 672; epoch: 23; loss: 0.05370887; min_loss: 0.05370887; since_best: 0; dice: 0.9596281608464776; 
    FastEstimator-Train: step: 680; loss: 0.031742807; steps/sec: 2.15; 
    FastEstimator-Train: step: 700; epoch: 24; epoch_time: 12.16 sec; 
    FastEstimator-Eval: step: 700; epoch: 24; loss: 0.054277744; min_loss: 0.05370887; since_best: 1; dice: 0.9582266396901049; 
    FastEstimator-Finish: step: 700; total_time: 392.64 sec; lung_segmentation_lr: 0.0001; 


# Inferencing

Let's visualize the prediction from the neural network. We select a radom image from the dataset.


```python
image_path = df['image'].sample(random_state=3).values[0]
```

## Pass Image through Pipeline and Network

We create a data dict, and call `Pipeline.transform()`.


```python
data = {'image': image_path}
data = pipeline.transform(data, mode="infer")
```

After the pipeline, we rebuild the model by providing the trained weights path and a new network.


```python
weights_path = os.path.join(save_dir, "lung_segmentation_best_Dice.pt") # your model_path

model = fe.build(model_fn=lambda: UNet(input_size=(1, 512, 512)),
                 optimizer_fn=lambda x: torch.optim.Adam(params=x, lr=0.0001),
                 model_names="lung_segmentation",
                 weights_path=weights_path)
```

    Loaded model weights from /tmp/tmpsqurxgnr/lung_segmentation_best_dice.pt



```python
network = fe.Network(ops=[ModelOp(inputs="image", model=model, outputs="pred_segment")])
```

We call `Network.transform()` to get outputs from our network.


```python
pred = network.transform(data, mode="infer")
```

## Visualize Outputs


```python
img = np.squeeze(pred['image'].numpy())
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_rgb = (img_rgb * 255).astype(np.uint8)
```


```python
mask = pred['pred_segment'].numpy()
mask = np.squeeze(mask)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
mask_rgb = (mask_rgb * 255).astype(np.uint8)
```


```python
_, mask_thres = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
mask_overlay = mask_rgb * np.expand_dims(mask_thres, axis=-1)
mask_overlay = np.where(mask_overlay != [0, 0, 0], [255, 0, 0], [0, 0, 0])
mask_overlay = mask_overlay.astype(np.uint8)
img_with_mask = cv2.addWeighted(img_rgb, 0.7, mask_overlay, 0.3, 0)
maskgt_with_maskpred = cv2.addWeighted(mask_rgb, 0.7, mask_overlay, 0.3, 0)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax[0].imshow(img_rgb)
ax[0].set_title('original lung')
ax[1].imshow(img_with_mask)
ax[1].set_title('predict mask ')
plt.show()
```


![png](assets/example/semantic_segmentation/unet_files/unet_40_0.png)

