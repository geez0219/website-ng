# Breast cancer detection example

Import the required libraries


```python
import tempfile

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

import fastestimator as fe
from fastestimator.dataset.data import breast_cancer
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy
```


```python
#training parameters
batch_size = 4
epochs = 10
save_dir = tempfile.mkdtemp()
max_steps_per_epoch = None
```

# Download data

This downloads the tabular data with different features stored in numerical format as a table. We then split the data into train, test and eval data sets.


```python
train_data, eval_data = breast_cancer.load_data()
test_data = eval_data.split(0.5)
```

This is what the raw data looks like:


```python
df = pd.DataFrame.from_dict(train_data.data, orient='index')
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[9.029, 17.33, 58.79, 250.5, 0.1066, 0.1413, 0...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[21.09, 26.57, 142.7, 1311.0, 0.1141, 0.2832, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[9.173, 13.86, 59.2, 260.9, 0.07721, 0.08751, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[10.65, 25.22, 68.01, 347.0, 0.09657, 0.07234,...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[10.17, 14.88, 64.55, 311.9, 0.1134, 0.08061, ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
scaler = StandardScaler()
train_data["x"] = scaler.fit_transform(train_data["x"])
eval_data["x"] = scaler.transform(eval_data["x"])
test_data["x"] = scaler.transform(test_data["x"])
```

# Building Components

## Step 1: Create Pipeline

We create the pipeline with the usual train, test and eval data alongwith the batch size


```python
pipeline = fe.Pipeline(train_data=train_data, eval_data=eval_data, test_data=test_data, batch_size=batch_size)
```

## Step 2: Create Network

We first define the network in a function that can later be passed on to the network


```python
def create_dnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation="relu", input_shape=(30, )))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model
```


```python
model = fe.build(model_fn=create_dnn, optimizer_fn="adam")
network = fe.Network(ops=[
    ModelOp(inputs="x", model=model, outputs="y_pred"),
    CrossEntropy(inputs=("y_pred", "y"), outputs="ce"),
    UpdateOp(model=model, loss_name="ce", mode="!infer")
])
```

## Step 3: Create Estimator


```python
traces = [
    Accuracy(true_key="y", pred_key="y_pred"),
    BestModelSaver(model=model, save_dir=save_dir, metric="accuracy", save_best_mode="max")
]
estimator = fe.Estimator(pipeline=pipeline,
                         network=network,
                         epochs=epochs,
                         log_steps=10,
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
                                                                            
    
    FastEstimator-Start: step: 1; model_lr: 0.001; 
    FastEstimator-Train: step: 1; ce: 0.58930933; 
    FastEstimator-Train: step: 10; ce: 1.2191963; steps/sec: 342.02; 
    FastEstimator-Train: step: 20; ce: 0.6330318; steps/sec: 422.95; 
    FastEstimator-Train: step: 30; ce: 0.68403095; steps/sec: 400.86; 
    FastEstimator-Train: step: 40; ce: 0.70622563; steps/sec: 277.93; 
    FastEstimator-Train: step: 50; ce: 0.7649698; steps/sec: 443.68; 
    FastEstimator-Train: step: 60; ce: 0.70189; steps/sec: 455.18; 
    FastEstimator-Train: step: 70; ce: 0.6120157; steps/sec: 486.19; 
    FastEstimator-Train: step: 80; ce: 0.6461396; steps/sec: 495.01; 
    FastEstimator-Train: step: 90; ce: 0.709924; steps/sec: 397.38; 
    FastEstimator-Train: step: 100; ce: 0.69695604; steps/sec: 545.18; 
    FastEstimator-Train: step: 110; ce: 0.5406225; steps/sec: 587.24; 
    FastEstimator-Train: step: 114; epoch: 1; epoch_time: 3.3 sec; 
    FastEstimator-ModelSaver: saved model to ./model_best_accuracy.h5
    FastEstimator-Eval: step: 114; epoch: 1; ce: 0.49621966; min_ce: 0.49621966; since_best: 0; accuracy: 0.9824561403508771; 
    FastEstimator-Train: step: 120; ce: 0.55340487; steps/sec: 7.94; 
    FastEstimator-Train: step: 130; ce: 0.31839967; steps/sec: 308.44; 
    FastEstimator-Train: step: 140; ce: 0.16889682; steps/sec: 482.01; 
    FastEstimator-Train: step: 150; ce: 0.5158031; steps/sec: 407.94; 
    FastEstimator-Train: step: 160; ce: 0.6378304; steps/sec: 386.75; 
    FastEstimator-Train: step: 170; ce: 1.1647241; steps/sec: 447.16; 
    FastEstimator-Train: step: 180; ce: 0.5274984; steps/sec: 500.64; 
    FastEstimator-Train: step: 190; ce: 0.68258667; steps/sec: 481.22; 
    FastEstimator-Train: step: 200; ce: 0.35559005; steps/sec: 476.14; 
    FastEstimator-Train: step: 210; ce: 0.46034247; steps/sec: 508.85; 
    FastEstimator-Train: step: 220; ce: 0.95580393; steps/sec: 548.84; 
    FastEstimator-Train: step: 228; epoch: 2; epoch_time: 1.39 sec; 
    FastEstimator-Eval: step: 228; epoch: 2; ce: 0.3193086; min_ce: 0.3193086; since_best: 0; accuracy: 0.9824561403508771; 
    FastEstimator-Train: step: 230; ce: 0.33260575; steps/sec: 8.49; 
    FastEstimator-Train: step: 240; ce: 0.2510308; steps/sec: 232.08; 
    FastEstimator-Train: step: 250; ce: 0.2878321; steps/sec: 666.82; 
    FastEstimator-Train: step: 260; ce: 0.1154226; steps/sec: 368.11; 
    FastEstimator-Train: step: 270; ce: 0.26300237; steps/sec: 414.99; 
    FastEstimator-Train: step: 280; ce: 0.5653368; steps/sec: 421.39; 
    FastEstimator-Train: step: 290; ce: 0.5872185; steps/sec: 402.68; 
    FastEstimator-Train: step: 300; ce: 0.27621573; steps/sec: 440.24; 
    FastEstimator-Train: step: 310; ce: 0.5477217; steps/sec: 481.69; 
    FastEstimator-Train: step: 320; ce: 0.4602429; steps/sec: 398.85; 
    FastEstimator-Train: step: 330; ce: 0.38244748; steps/sec: 546.57; 
    FastEstimator-Train: step: 340; ce: 0.5337428; steps/sec: 571.02; 
    FastEstimator-Train: step: 342; epoch: 3; epoch_time: 1.42 sec; 
    FastEstimator-Eval: step: 342; epoch: 3; ce: 0.18308732; min_ce: 0.18308732; since_best: 0; accuracy: 0.9824561403508771; 
    FastEstimator-Train: step: 350; ce: 0.13466343; steps/sec: 8.53; 
    FastEstimator-Train: step: 360; ce: 0.22628057; steps/sec: 368.34; 
    FastEstimator-Train: step: 370; ce: 0.5836228; steps/sec: 485.06; 
    FastEstimator-Train: step: 380; ce: 0.37300625; steps/sec: 409.87; 
    FastEstimator-Train: step: 390; ce: 0.2717349; steps/sec: 413.59; 
    FastEstimator-Train: step: 400; ce: 0.07554119; steps/sec: 433.84; 
    FastEstimator-Train: step: 410; ce: 0.20552614; steps/sec: 439.36; 
    FastEstimator-Train: step: 420; ce: 0.28509304; steps/sec: 448.96; 
    FastEstimator-Train: step: 430; ce: 0.32158756; steps/sec: 492.58; 
    FastEstimator-Train: step: 440; ce: 1.1102628; steps/sec: 525.49; 
    FastEstimator-Train: step: 450; ce: 0.31964102; steps/sec: 548.06; 
    FastEstimator-Train: step: 456; epoch: 4; epoch_time: 1.4 sec; 
    FastEstimator-ModelSaver: saved model to ./model_best_accuracy.h5
    FastEstimator-Eval: step: 456; epoch: 4; ce: 0.105911165; min_ce: 0.105911165; since_best: 0; accuracy: 1.0; 
    FastEstimator-Train: step: 460; ce: 0.4391592; steps/sec: 8.37; 
    FastEstimator-Train: step: 470; ce: 0.29870045; steps/sec: 297.42; 
    FastEstimator-Train: step: 480; ce: 0.03247342; steps/sec: 597.74; 
    FastEstimator-Train: step: 490; ce: 0.13323224; steps/sec: 393.92; 
    FastEstimator-Train: step: 500; ce: 0.58429027; steps/sec: 405.0; 
    FastEstimator-Train: step: 510; ce: 0.2376658; steps/sec: 455.97; 
    FastEstimator-Train: step: 520; ce: 0.4150503; steps/sec: 424.88; 
    FastEstimator-Train: step: 530; ce: 0.22695109; steps/sec: 451.62; 
    FastEstimator-Train: step: 540; ce: 0.42051294; steps/sec: 402.12; 
    FastEstimator-Train: step: 550; ce: 0.17364319; steps/sec: 389.83; 
    FastEstimator-Train: step: 560; ce: 0.06320181; steps/sec: 466.97; 
    FastEstimator-Train: step: 570; ce: 0.13996354; steps/sec: 518.8; 
    FastEstimator-Train: step: 570; epoch: 5; epoch_time: 1.46 sec; 
    FastEstimator-Eval: step: 570; epoch: 5; ce: 0.066059396; min_ce: 0.066059396; since_best: 0; accuracy: 1.0; 
    FastEstimator-Train: step: 580; ce: 0.12985338; steps/sec: 8.17; 
    FastEstimator-Train: step: 590; ce: 0.6419388; steps/sec: 373.15; 
    FastEstimator-Train: step: 600; ce: 0.2857446; steps/sec: 404.92; 
    FastEstimator-Train: step: 610; ce: 0.21400735; steps/sec: 381.65; 
    FastEstimator-Train: step: 620; ce: 0.27899668; steps/sec: 394.87; 
    FastEstimator-Train: step: 630; ce: 0.31599885; steps/sec: 472.31; 
    FastEstimator-Train: step: 640; ce: 0.036415085; steps/sec: 457.09; 
    FastEstimator-Train: step: 650; ce: 0.10052729; steps/sec: 461.82; 
    FastEstimator-Train: step: 660; ce: 0.40688303; steps/sec: 474.46; 
    FastEstimator-Train: step: 670; ce: 0.40816957; steps/sec: 517.75; 
    FastEstimator-Train: step: 680; ce: 0.40120217; steps/sec: 555.53; 
    FastEstimator-Train: step: 684; epoch: 6; epoch_time: 1.44 sec; 
    FastEstimator-Eval: step: 684; epoch: 6; ce: 0.04396173; min_ce: 0.04396173; since_best: 0; accuracy: 1.0; 
    FastEstimator-Train: step: 690; ce: 0.20741543; steps/sec: 8.33; 
    FastEstimator-Train: step: 700; ce: 0.12485474; steps/sec: 324.64; 
    FastEstimator-Train: step: 710; ce: 2.8970864e-05; steps/sec: 534.71; 
    FastEstimator-Train: step: 720; ce: 0.110491954; steps/sec: 402.98; 
    FastEstimator-Train: step: 730; ce: 0.10486858; steps/sec: 432.34; 
    FastEstimator-Train: step: 740; ce: 0.2951797; steps/sec: 421.45; 
    FastEstimator-Train: step: 750; ce: 0.65293443; steps/sec: 433.71; 
    FastEstimator-Train: step: 760; ce: 0.32570755; steps/sec: 461.43; 
    FastEstimator-Train: step: 770; ce: 0.35400242; steps/sec: 433.46; 
    FastEstimator-Train: step: 780; ce: 0.023054674; steps/sec: 483.0; 
    FastEstimator-Train: step: 790; ce: 0.16433364; steps/sec: 540.17; 
    FastEstimator-Train: step: 798; epoch: 7; epoch_time: 1.43 sec; 
    FastEstimator-Eval: step: 798; epoch: 7; ce: 0.040205613; min_ce: 0.040205613; since_best: 0; accuracy: 1.0; 
    FastEstimator-Train: step: 800; ce: 0.42427045; steps/sec: 8.52; 
    FastEstimator-Train: step: 810; ce: 0.39827985; steps/sec: 266.34; 
    FastEstimator-Train: step: 820; ce: 0.43165076; steps/sec: 775.07; 
    FastEstimator-Train: step: 830; ce: 0.06976031; steps/sec: 412.9; 
    FastEstimator-Train: step: 840; ce: 0.37039524; steps/sec: 441.05; 
    FastEstimator-Train: step: 850; ce: 0.10960688; steps/sec: 418.01; 
    FastEstimator-Train: step: 860; ce: 0.0070317476; steps/sec: 450.1; 
    FastEstimator-Train: step: 870; ce: 0.020452987; steps/sec: 434.06; 
    FastEstimator-Train: step: 880; ce: 0.12914097; steps/sec: 476.49; 
    FastEstimator-Train: step: 890; ce: 0.25528443; steps/sec: 466.42; 
    FastEstimator-Train: step: 900; ce: 0.18017673; steps/sec: 549.95; 
    FastEstimator-Train: step: 910; ce: 0.31777602; steps/sec: 583.67; 
    FastEstimator-Train: step: 912; epoch: 8; epoch_time: 1.41 sec; 
    FastEstimator-Eval: step: 912; epoch: 8; ce: 0.028554583; min_ce: 0.028554583; since_best: 0; accuracy: 1.0; 
    FastEstimator-Train: step: 920; ce: 0.24684253; steps/sec: 8.42; 
    FastEstimator-Train: step: 930; ce: 0.19438684; steps/sec: 365.05; 
    FastEstimator-Train: step: 940; ce: 0.1568121; steps/sec: 477.85; 
    FastEstimator-Train: step: 950; ce: 0.3368427; steps/sec: 371.39; 
    FastEstimator-Train: step: 960; ce: 0.20518681; steps/sec: 411.72; 
    FastEstimator-Train: step: 970; ce: 0.13320616; steps/sec: 401.91; 
    FastEstimator-Train: step: 980; ce: 0.1800138; steps/sec: 470.79; 
    FastEstimator-Train: step: 990; ce: 0.10868286; steps/sec: 421.41; 
    FastEstimator-Train: step: 1000; ce: 0.040300086; steps/sec: 467.66; 
    FastEstimator-Train: step: 1010; ce: 0.42622733; steps/sec: 505.31; 
    FastEstimator-Train: step: 1020; ce: 0.06701453; steps/sec: 530.27; 
    FastEstimator-Train: step: 1026; epoch: 9; epoch_time: 1.42 sec; 
    FastEstimator-Eval: step: 1026; epoch: 9; ce: 0.019402837; min_ce: 0.019402837; since_best: 0; accuracy: 1.0; 
    FastEstimator-Train: step: 1030; ce: 0.27714887; steps/sec: 8.63; 
    FastEstimator-Train: step: 1040; ce: 0.074241355; steps/sec: 302.11; 
    FastEstimator-Train: step: 1050; ce: 0.025415465; steps/sec: 640.07; 
    FastEstimator-Train: step: 1060; ce: 0.21693969; steps/sec: 447.73; 
    FastEstimator-Train: step: 1070; ce: 0.120441705; steps/sec: 432.52; 
    FastEstimator-Train: step: 1080; ce: 0.25360084; steps/sec: 459.08; 
    FastEstimator-Train: step: 1090; ce: 0.22401881; steps/sec: 493.29; 
    FastEstimator-Train: step: 1100; ce: 0.112028226; steps/sec: 485.33; 
    FastEstimator-Train: step: 1110; ce: 2.4293017; steps/sec: 446.65; 
    FastEstimator-Train: step: 1120; ce: 0.19810674; steps/sec: 514.16; 
    FastEstimator-Train: step: 1130; ce: 0.12599353; steps/sec: 529.87; 
    FastEstimator-Train: step: 1140; ce: 0.23468983; steps/sec: 580.97; 
    FastEstimator-Train: step: 1140; epoch: 10; epoch_time: 1.38 sec; 
    FastEstimator-Eval: step: 1140; epoch: 10; ce: 0.017586827; min_ce: 0.017586827; since_best: 0; accuracy: 1.0; 
    FastEstimator-Finish: step: 1140; total_time: 28.21 sec; model_lr: 0.001; 


## Model testing
`Estimator.test` triggers model testing with test dataset that specified in `Pipeline`. We can evaluate the model performance in the classification accuracy. 


```python
estimator.test()
```

    FastEstimator-Test: epoch: 10; accuracy: 0.9649122807017544; 

