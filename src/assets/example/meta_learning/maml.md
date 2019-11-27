# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

In this notebook, we will demostrate how to implement *Model-Agnostic Meta-Learning (MAML)* in FastEstimator.
We will train a simple sinusoid regression model using *MAML* for a 10-shot regression task.
The objective of *MAML* is to learn a model that is trained on a larger set of related tasks so that the trained model can quickly adapt to the given task.


```python
import numpy as np
import tensorflow as tf
import fastestimator as fe
```


```python
from tensorflow.keras import layers, losses
from tensorflow.keras import Sequential, Model
```

## Step 1: Create Data Pipeline

First, we will create data pipeline that will yield a batch of sinusoid regression tasks where each task yields 10 data points of *x* and *y* defined by the following equation:

$y = amplitude * sin(x + phase)$
where $amplitude$, $phase$, and $x$ are randomly chosen for each task.

The dimension for each batch is *(number of task, number of sample, 1)*


```python
def generate_random_sine(amp_range=[0.1, 5.0], phase_range=[0, np.pi], x_range=[-5.0, 5.0], K=10):
    while True:
        a = np.random.uniform(amp_range[0], amp_range[1])
        b = np.random.uniform(phase_range[0], phase_range[1])
        x = np.random.uniform(x_range[0], x_range[1], 2*K).astype(np.float32)
        y = a * np.sin(x + b).astype(np.float32)
        yield {"x_meta_train": np.expand_dims(x[:K], axis=-1),
               "x_meta_test": np.expand_dims(x[K:], axis=-1), 
               "y_meta_train": np.expand_dims(y[:K], axis=-1),
               "y_meta_test": np.expand_dims(y[K:], axis=-1),
               "amp":a, 
               "phase":b}
```


```python
sine_test_data = next(generate_random_sine())
```


```python
pipeline = fe.Pipeline(
    data={"train":generate_random_sine},
    batch_size=25
)
```

## Step 2: Define Network
For the regression task, we use a simple Multi-Layer Perceptron (MLP) as described in the paper.


```python
def build_sine_model():
    mdl = Sequential()
    mdl.add(layers.Dense(40, input_shape=(1,), activation="relu"))
    mdl.add(layers.Dense(40, activation="relu"))
    mdl.add(layers.Dense(1))
    return mdl


meta_model = fe.build(
    model_def=build_sine_model,
    model_name="meta_model",
    loss_name="meta_loss",
    optimizer="adam"
)
```

### Define Operators
The training scheme of *MAML* is as follows:

Given a meta train batch $D^{tr}$, a meta test batch $D^{test}$, and a meta model weight $\theta$,
1. Compute $\mathcal{L}\left(D^{tr}, \theta\right)$
2. Compute $\nabla_{\theta}\mathcal{L}\left(D^{tr}, \theta\right)$
3. Define a tentative model weight $\phi$ as $\theta - \beta\nabla_{\theta}\mathcal{L}\left(D^{tr}, \theta\right)$ where $\beta$ is an inner learning rate.
4. Compute $\mathcal{L}\left(D^{test}, \phi\right)$
5. Compute $\nabla_{\theta}\mathcal{L}\left(D^{test}, \phi\right)$

Note that the gradient computed in step 5 is the second order gradient because
$\nabla_{\theta}\mathcal{L}\left(D^{test}, \phi\right) = \nabla_{\theta}\mathcal{L}\left(D^{test}, \theta - \beta\nabla_{\theta}\mathcal{L}\left(D^{tr}, \theta\right)\right)$

We have to define the following operators to support the meta learning:
* MetaModelOp to apply *ModelOp* to each task
* MetaMSE to apply *MeanSquaredError* to each task
* InnerGradientOp to compute the first order gradient for each task
* InnerUpdateOp to obtain $\phi$
* MetaForwardOp to compute the forward pass of $\phi$ on $D^{test}$


```python
from fastestimator.op import TensorOp
from fastestimator.op.tensorop import ModelOp
from fastestimator.op.tensorop import UpdateOp, Gradients
from fastestimator.op.tensorop.loss import MeanSquaredError, Loss        

class MetaModelOp(ModelOp):
    def _single_forward(self, data):
        return self.model(data, training=True)
    
    def forward(self, data, state):        
        out = tf.map_fn(fn=self._single_forward, elems=data, dtype=tf.float32)
        return out
    
class MetaMSE(MeanSquaredError):
    def forward(self, data, state):
        true, pred = data
        out = self.loss_obj(true, pred)
        return tf.reduce_mean(out, axis=1)

    
class InnerGradientOp(TensorOp):
    def __init__(self, loss, model, outputs):
        super().__init__(inputs=loss, outputs=outputs, mode="train")
        self.model = model
        
    def forward(self, data, state):
        loss = data
        tape = state['tape']
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients, self.model.trainable_variables     

class InnerUpdateOp(TensorOp):
    def __init__(self, inputs, outputs, inner_lr):
        super().__init__(inputs=inputs, outputs=outputs, mode="train")
        self.inner_lr = inner_lr
        
    def forward(self, data, state):
        g, v = data
        return [v_ - self.inner_lr * g_ for g_, v_ in zip(g, v)]

class MetaForwardOp(TensorOp):
    def forward(self, data, state):
        x0, model_var = data
        
        def _single_forward(x):
            out = tf.nn.relu(tf.matmul(x, model_var[0]) + model_var[1])
            for i in range(2, len(model_var) - 2, 2):
                out = tf.nn.relu(tf.matmul(out, model_var[2]) + model_var[3])        
            out = tf.matmul(out, model_var[-2]) + model_var[-1]
            return out
        
        return tf.map_fn(_single_forward, elems=x0, dtype=tf.float32)
```

Given these operators, we need to put everything together in *Network*.


```python
network = fe.Network(
    ops=[
        MetaModelOp(inputs="x_meta_train", outputs="y_meta_pred", model=meta_model),
        MetaMSE(inputs=("y_meta_train", "y_meta_pred"), outputs="inner_loss"),
        InnerGradientOp(loss="inner_loss", model=meta_model, outputs=("inner_grad", "model_var")),
        InnerUpdateOp(inputs=("inner_grad", "model_var"), outputs="model_var", inner_lr=1e-3),
        MetaForwardOp(inputs=("x_meta_test", "model_var"), outputs="y_pred"),
        MetaMSE(inputs=("y_meta_test", "y_pred"), outputs="meta_loss"),
        Gradients(loss="meta_loss", models=meta_model, outputs="meta_grad"),
        UpdateOp(model=meta_model, gradients="meta_grad")
    ]
)
```

## Step 3: Define Estimator
Given *Pipeline* and *Network*, we put these together in *Estimator*


```python
estimator = fe.Estimator(
    network=network,
    pipeline=pipeline,
    epochs=1,
    steps_per_epoch=20000
)
```


```python
estimator.fit()
```

        ______           __  ______     __  _                 __            
       / ____/___ ______/ /_/ ____/____/ /_(_)___ ___  ____ _/ /_____  _____
      / /_  / __ `/ ___/ __/ __/ / ___/ __/ / __ `__ \/ __ `/ __/ __ \/ ___/
     / __/ / /_/ (__  ) /_/ /___(__  ) /_/ / / / / / / /_/ / /_/ /_/ / /    
    /_/    \__,_/____/\__/_____/____/\__/_/_/ /_/ /_/\__,_/\__/\____/_/     
                                                                            
    
    WARNING:tensorflow:Calling GradientTape.gradient on a persistent tape inside its context is significantly less efficient than calling it outside the context (it causes the gradient ops to be recorded on the tape, leading to increased CPU and memory usage). Only call GradientTape.gradient inside the context if you actually want to trace the gradient in order to compute higher order derivatives.
    FastEstimator-Warn: No ModelSaver Trace detected. Models will not be saved.
    FastEstimator-Start: step: 0; total_train_steps: 20000; meta_model_lr: 0.001; 
    WARNING:tensorflow:Calling GradientTape.gradient on a persistent tape inside its context is significantly less efficient than calling it outside the context (it causes the gradient ops to be recorded on the tape, leading to increased CPU and memory usage). Only call GradientTape.gradient inside the context if you actually want to trace the gradient in order to compute higher order derivatives.
    FastEstimator-Train: step: 0; meta_loss: 5.723891; inner_loss: 6.9464946; 
    FastEstimator-Train: step: 100; meta_loss: 3.4250093; inner_loss: 2.9725184; examples/sec: 370.3; progress: 0.5%; 
    FastEstimator-Train: step: 200; meta_loss: 3.6458304; inner_loss: 4.013458; examples/sec: 377.7; progress: 1.0%; 
    FastEstimator-Train: step: 300; meta_loss: 3.0723948; inner_loss: 3.707258; examples/sec: 377.7; progress: 1.5%; 
    FastEstimator-Train: step: 400; meta_loss: 3.0435414; inner_loss: 2.8360772; examples/sec: 382.4; progress: 2.0%; 
    FastEstimator-Train: step: 500; meta_loss: 2.6957202; inner_loss: 3.2137213; examples/sec: 380.5; progress: 2.5%; 
    FastEstimator-Train: step: 600; meta_loss: 2.14638; inner_loss: 2.26405; examples/sec: 380.8; progress: 3.0%; 
    FastEstimator-Train: step: 700; meta_loss: 2.8416154; inner_loss: 3.3307781; examples/sec: 378.7; progress: 3.5%; 
    FastEstimator-Train: step: 800; meta_loss: 2.9478874; inner_loss: 3.4892333; examples/sec: 379.5; progress: 4.0%; 
    FastEstimator-Train: step: 900; meta_loss: 2.3352726; inner_loss: 2.1838467; examples/sec: 382.6; progress: 4.5%; 
    FastEstimator-Train: step: 1000; meta_loss: 2.4479403; inner_loss: 2.702093; examples/sec: 374.9; progress: 5.0%; 
    FastEstimator-Train: step: 1100; meta_loss: 3.4396672; inner_loss: 3.1015794; examples/sec: 374.1; progress: 5.5%; 
    FastEstimator-Train: step: 1200; meta_loss: 2.8182912; inner_loss: 2.9433377; examples/sec: 372.2; progress: 6.0%; 
    FastEstimator-Train: step: 1300; meta_loss: 2.5766463; inner_loss: 2.8751929; examples/sec: 371.9; progress: 6.5%; 
    FastEstimator-Train: step: 1400; meta_loss: 3.2975922; inner_loss: 3.3123927; examples/sec: 373.3; progress: 7.0%; 
    FastEstimator-Train: step: 1500; meta_loss: 3.0313966; inner_loss: 3.2418108; examples/sec: 371.0; progress: 7.5%; 
    FastEstimator-Train: step: 1600; meta_loss: 3.6314025; inner_loss: 3.9095612; examples/sec: 370.9; progress: 8.0%; 
    FastEstimator-Train: step: 1700; meta_loss: 2.3394818; inner_loss: 2.532619; examples/sec: 369.4; progress: 8.5%; 
    FastEstimator-Train: step: 1800; meta_loss: 3.0216534; inner_loss: 3.0520546; examples/sec: 368.8; progress: 9.0%; 
    FastEstimator-Train: step: 1900; meta_loss: 3.0035326; inner_loss: 3.0798998; examples/sec: 370.8; progress: 9.5%; 
    FastEstimator-Train: step: 2000; meta_loss: 3.69296; inner_loss: 3.859979; examples/sec: 376.1; progress: 10.0%; 
    FastEstimator-Train: step: 2100; meta_loss: 2.3376408; inner_loss: 2.624551; examples/sec: 379.5; progress: 10.5%; 
    FastEstimator-Train: step: 2200; meta_loss: 3.0599577; inner_loss: 2.9528558; examples/sec: 378.4; progress: 11.0%; 
    FastEstimator-Train: step: 2300; meta_loss: 3.07903; inner_loss: 3.075372; examples/sec: 379.5; progress: 11.5%; 
    FastEstimator-Train: step: 2400; meta_loss: 3.2811193; inner_loss: 2.8805897; examples/sec: 376.7; progress: 12.0%; 
    FastEstimator-Train: step: 2500; meta_loss: 2.89411; inner_loss: 2.7017875; examples/sec: 376.7; progress: 12.5%; 
    FastEstimator-Train: step: 2600; meta_loss: 3.5004468; inner_loss: 3.273705; examples/sec: 378.2; progress: 13.0%; 
    FastEstimator-Train: step: 2700; meta_loss: 2.9885342; inner_loss: 3.4973888; examples/sec: 378.7; progress: 13.5%; 
    FastEstimator-Train: step: 2800; meta_loss: 3.015358; inner_loss: 3.0963686; examples/sec: 377.2; progress: 14.0%; 
    FastEstimator-Train: step: 2900; meta_loss: 3.0528479; inner_loss: 4.1257095; examples/sec: 376.1; progress: 14.5%; 
    FastEstimator-Train: step: 3000; meta_loss: 2.3670251; inner_loss: 2.335838; examples/sec: 376.8; progress: 15.0%; 
    FastEstimator-Train: step: 3100; meta_loss: 2.3240917; inner_loss: 2.281818; examples/sec: 377.9; progress: 15.5%; 
    FastEstimator-Train: step: 3200; meta_loss: 2.9467633; inner_loss: 2.800916; examples/sec: 374.6; progress: 16.0%; 
    FastEstimator-Train: step: 3300; meta_loss: 3.1842215; inner_loss: 2.9574022; examples/sec: 377.4; progress: 16.5%; 
    FastEstimator-Train: step: 3400; meta_loss: 2.562764; inner_loss: 2.4471729; examples/sec: 377.2; progress: 17.0%; 
    FastEstimator-Train: step: 3500; meta_loss: 3.0540297; inner_loss: 3.1730306; examples/sec: 376.6; progress: 17.5%; 
    FastEstimator-Train: step: 3600; meta_loss: 3.067585; inner_loss: 3.4638128; examples/sec: 376.6; progress: 18.0%; 
    FastEstimator-Train: step: 3700; meta_loss: 3.0378575; inner_loss: 3.3765068; examples/sec: 377.6; progress: 18.5%; 
    FastEstimator-Train: step: 3800; meta_loss: 2.230997; inner_loss: 2.3409219; examples/sec: 378.3; progress: 19.0%; 
    FastEstimator-Train: step: 3900; meta_loss: 2.4442446; inner_loss: 2.1866262; examples/sec: 377.3; progress: 19.5%; 
    FastEstimator-Train: step: 4000; meta_loss: 2.9056315; inner_loss: 2.8474798; examples/sec: 377.9; progress: 20.0%; 
    FastEstimator-Train: step: 4100; meta_loss: 2.8586812; inner_loss: 3.199026; examples/sec: 377.1; progress: 20.5%; 
    FastEstimator-Train: step: 4200; meta_loss: 2.6487834; inner_loss: 2.863139; examples/sec: 372.8; progress: 21.0%; 
    FastEstimator-Train: step: 4300; meta_loss: 4.3500257; inner_loss: 4.258764; examples/sec: 372.8; progress: 21.5%; 
    FastEstimator-Train: step: 4400; meta_loss: 2.0445044; inner_loss: 2.291754; examples/sec: 371.7; progress: 22.0%; 
    FastEstimator-Train: step: 4500; meta_loss: 3.142999; inner_loss: 3.5001268; examples/sec: 373.6; progress: 22.5%; 
    FastEstimator-Train: step: 4600; meta_loss: 2.754345; inner_loss: 3.1272655; examples/sec: 378.0; progress: 23.0%; 
    FastEstimator-Train: step: 4700; meta_loss: 3.433843; inner_loss: 3.5478616; examples/sec: 376.5; progress: 23.5%; 
    FastEstimator-Train: step: 4800; meta_loss: 2.6278224; inner_loss: 2.4901352; examples/sec: 378.6; progress: 24.0%; 
    FastEstimator-Train: step: 4900; meta_loss: 2.7712255; inner_loss: 2.622183; examples/sec: 381.4; progress: 24.5%; 
    FastEstimator-Train: step: 5000; meta_loss: 3.5073233; inner_loss: 3.1740746; examples/sec: 373.9; progress: 25.0%; 
    FastEstimator-Train: step: 5100; meta_loss: 1.8426422; inner_loss: 2.2200181; examples/sec: 377.7; progress: 25.5%; 
    FastEstimator-Train: step: 5200; meta_loss: 3.5395603; inner_loss: 3.805974; examples/sec: 376.3; progress: 26.0%; 
    FastEstimator-Train: step: 5300; meta_loss: 2.2879512; inner_loss: 2.3388176; examples/sec: 372.0; progress: 26.5%; 
    FastEstimator-Train: step: 5400; meta_loss: 3.1373749; inner_loss: 3.2174873; examples/sec: 372.9; progress: 27.0%; 
    FastEstimator-Train: step: 5500; meta_loss: 2.609693; inner_loss: 2.8761358; examples/sec: 372.5; progress: 27.5%; 
    FastEstimator-Train: step: 5600; meta_loss: 4.603827; inner_loss: 4.809171; examples/sec: 378.4; progress: 28.0%; 
    FastEstimator-Train: step: 5700; meta_loss: 3.0636597; inner_loss: 3.1069462; examples/sec: 378.5; progress: 28.5%; 
    FastEstimator-Train: step: 5800; meta_loss: 2.235212; inner_loss: 2.4183726; examples/sec: 376.9; progress: 29.0%; 
    FastEstimator-Train: step: 5900; meta_loss: 2.9981494; inner_loss: 2.952943; examples/sec: 381.5; progress: 29.5%; 
    FastEstimator-Train: step: 6000; meta_loss: 1.610773; inner_loss: 1.94066; examples/sec: 379.5; progress: 30.0%; 
    FastEstimator-Train: step: 6100; meta_loss: 2.3705523; inner_loss: 2.6015983; examples/sec: 377.6; progress: 30.5%; 
    FastEstimator-Train: step: 6200; meta_loss: 3.6479864; inner_loss: 3.4792807; examples/sec: 379.4; progress: 31.0%; 
    FastEstimator-Train: step: 6300; meta_loss: 2.3008645; inner_loss: 2.7936928; examples/sec: 378.6; progress: 31.5%; 
    FastEstimator-Train: step: 6400; meta_loss: 2.5303686; inner_loss: 2.9379578; examples/sec: 375.0; progress: 32.0%; 
    FastEstimator-Train: step: 6500; meta_loss: 2.9302416; inner_loss: 3.420107; examples/sec: 371.2; progress: 32.5%; 
    FastEstimator-Train: step: 6600; meta_loss: 3.4646788; inner_loss: 3.4846115; examples/sec: 372.1; progress: 33.0%; 
    FastEstimator-Train: step: 6700; meta_loss: 2.819759; inner_loss: 2.6378808; examples/sec: 371.4; progress: 33.5%; 
    FastEstimator-Train: step: 6800; meta_loss: 3.1392748; inner_loss: 3.1842782; examples/sec: 373.6; progress: 34.0%; 
    FastEstimator-Train: step: 6900; meta_loss: 2.193612; inner_loss: 2.4759164; examples/sec: 376.9; progress: 34.5%; 
    FastEstimator-Train: step: 7000; meta_loss: 2.4757478; inner_loss: 2.763305; examples/sec: 372.1; progress: 35.0%; 
    FastEstimator-Train: step: 7100; meta_loss: 2.9558313; inner_loss: 3.2005665; examples/sec: 369.6; progress: 35.5%; 
    FastEstimator-Train: step: 7200; meta_loss: 4.602907; inner_loss: 4.7396045; examples/sec: 367.7; progress: 36.0%; 
    FastEstimator-Train: step: 7300; meta_loss: 2.5914018; inner_loss: 2.5922923; examples/sec: 371.1; progress: 36.5%; 
    FastEstimator-Train: step: 7400; meta_loss: 2.476404; inner_loss: 2.7485769; examples/sec: 373.8; progress: 37.0%; 
    FastEstimator-Train: step: 7500; meta_loss: 2.1848364; inner_loss: 2.1454349; examples/sec: 378.5; progress: 37.5%; 
    FastEstimator-Train: step: 7600; meta_loss: 3.1205204; inner_loss: 2.9075384; examples/sec: 379.6; progress: 38.0%; 
    FastEstimator-Train: step: 7700; meta_loss: 3.0812776; inner_loss: 3.2400622; examples/sec: 380.3; progress: 38.5%; 
    FastEstimator-Train: step: 7800; meta_loss: 4.006408; inner_loss: 3.8411503; examples/sec: 378.4; progress: 39.0%; 
    FastEstimator-Train: step: 7900; meta_loss: 3.2471979; inner_loss: 3.3881464; examples/sec: 377.4; progress: 39.5%; 
    FastEstimator-Train: step: 8000; meta_loss: 2.6738243; inner_loss: 2.4311273; examples/sec: 376.5; progress: 40.0%; 
    FastEstimator-Train: step: 8100; meta_loss: 2.3496838; inner_loss: 2.3924034; examples/sec: 376.4; progress: 40.5%; 
    FastEstimator-Train: step: 8200; meta_loss: 1.9857275; inner_loss: 2.4718354; examples/sec: 377.7; progress: 41.0%; 
    FastEstimator-Train: step: 8300; meta_loss: 2.6773899; inner_loss: 2.8510091; examples/sec: 377.0; progress: 41.5%; 
    FastEstimator-Train: step: 8400; meta_loss: 2.5514374; inner_loss: 2.7643144; examples/sec: 376.7; progress: 42.0%; 
    FastEstimator-Train: step: 8500; meta_loss: 3.4369595; inner_loss: 3.3562915; examples/sec: 375.1; progress: 42.5%; 
    FastEstimator-Train: step: 8600; meta_loss: 3.7530208; inner_loss: 3.6764684; examples/sec: 377.1; progress: 43.0%; 
    FastEstimator-Train: step: 8700; meta_loss: 2.9853857; inner_loss: 3.520512; examples/sec: 376.5; progress: 43.5%; 
    FastEstimator-Train: step: 8800; meta_loss: 2.2675889; inner_loss: 2.2104526; examples/sec: 375.6; progress: 44.0%; 
    FastEstimator-Train: step: 8900; meta_loss: 4.2700458; inner_loss: 4.448406; examples/sec: 379.0; progress: 44.5%; 
    FastEstimator-Train: step: 9000; meta_loss: 3.4851327; inner_loss: 3.3740053; examples/sec: 380.1; progress: 45.0%; 
    FastEstimator-Train: step: 9100; meta_loss: 2.9665058; inner_loss: 3.030382; examples/sec: 378.0; progress: 45.5%; 
    FastEstimator-Train: step: 9200; meta_loss: 2.6753213; inner_loss: 2.7245958; examples/sec: 376.4; progress: 46.0%; 
    FastEstimator-Train: step: 9300; meta_loss: 2.3604999; inner_loss: 2.5478077; examples/sec: 376.8; progress: 46.5%; 
    FastEstimator-Train: step: 9400; meta_loss: 2.742097; inner_loss: 2.837412; examples/sec: 376.1; progress: 47.0%; 
    FastEstimator-Train: step: 9500; meta_loss: 3.2408192; inner_loss: 3.2863173; examples/sec: 371.1; progress: 47.5%; 
    FastEstimator-Train: step: 9600; meta_loss: 3.8287497; inner_loss: 3.6071265; examples/sec: 370.5; progress: 48.0%; 
    FastEstimator-Train: step: 9700; meta_loss: 3.7905853; inner_loss: 3.5408084; examples/sec: 372.3; progress: 48.5%; 
    FastEstimator-Train: step: 9800; meta_loss: 3.517132; inner_loss: 3.4375656; examples/sec: 377.9; progress: 49.0%; 
    FastEstimator-Train: step: 9900; meta_loss: 3.6941316; inner_loss: 3.8764467; examples/sec: 375.1; progress: 49.5%; 
    FastEstimator-Train: step: 10000; meta_loss: 2.8828595; inner_loss: 3.2284389; examples/sec: 373.6; progress: 50.0%; 
    FastEstimator-Train: step: 10100; meta_loss: 2.9444582; inner_loss: 2.9833598; examples/sec: 373.8; progress: 50.5%; 
    FastEstimator-Train: step: 10200; meta_loss: 3.5511413; inner_loss: 3.5146475; examples/sec: 374.3; progress: 51.0%; 
    FastEstimator-Train: step: 10300; meta_loss: 2.5076935; inner_loss: 3.3944893; examples/sec: 375.9; progress: 51.5%; 
    FastEstimator-Train: step: 10400; meta_loss: 3.2614484; inner_loss: 3.5075848; examples/sec: 377.9; progress: 52.0%; 
    FastEstimator-Train: step: 10500; meta_loss: 2.1804242; inner_loss: 2.2428102; examples/sec: 377.3; progress: 52.5%; 
    FastEstimator-Train: step: 10600; meta_loss: 2.39922; inner_loss: 3.3829205; examples/sec: 375.5; progress: 53.0%; 
    FastEstimator-Train: step: 10700; meta_loss: 2.7134447; inner_loss: 2.9140613; examples/sec: 373.5; progress: 53.5%; 
    FastEstimator-Train: step: 10800; meta_loss: 2.0532033; inner_loss: 2.0323362; examples/sec: 370.8; progress: 54.0%; 
    FastEstimator-Train: step: 10900; meta_loss: 2.093052; inner_loss: 2.4826322; examples/sec: 371.9; progress: 54.5%; 
    FastEstimator-Train: step: 11000; meta_loss: 2.5783112; inner_loss: 2.8034296; examples/sec: 377.2; progress: 55.0%; 
    FastEstimator-Train: step: 11100; meta_loss: 2.8563297; inner_loss: 3.5608532; examples/sec: 377.5; progress: 55.5%; 
    FastEstimator-Train: step: 11200; meta_loss: 2.3807635; inner_loss: 2.6072125; examples/sec: 378.5; progress: 56.0%; 
    FastEstimator-Train: step: 11300; meta_loss: 3.0397007; inner_loss: 3.0780365; examples/sec: 381.2; progress: 56.5%; 
    FastEstimator-Train: step: 11400; meta_loss: 3.9022155; inner_loss: 3.5584865; examples/sec: 377.6; progress: 57.0%; 
    FastEstimator-Train: step: 11500; meta_loss: 3.319961; inner_loss: 3.779936; examples/sec: 378.3; progress: 57.5%; 
    FastEstimator-Train: step: 11600; meta_loss: 2.797553; inner_loss: 2.6223083; examples/sec: 376.7; progress: 58.0%; 
    FastEstimator-Train: step: 11700; meta_loss: 3.6053052; inner_loss: 3.3402529; examples/sec: 379.1; progress: 58.5%; 
    FastEstimator-Train: step: 11800; meta_loss: 3.6330593; inner_loss: 3.8007; examples/sec: 377.7; progress: 59.0%; 
    FastEstimator-Train: step: 11900; meta_loss: 2.6815295; inner_loss: 3.0236065; examples/sec: 378.3; progress: 59.5%; 
    FastEstimator-Train: step: 12000; meta_loss: 3.4522667; inner_loss: 3.737786; examples/sec: 378.7; progress: 60.0%; 
    FastEstimator-Train: step: 12100; meta_loss: 2.9640062; inner_loss: 2.9199; examples/sec: 376.0; progress: 60.5%; 
    FastEstimator-Train: step: 12200; meta_loss: 3.2009375; inner_loss: 3.3087797; examples/sec: 380.8; progress: 61.0%; 
    FastEstimator-Train: step: 12300; meta_loss: 2.588437; inner_loss: 3.0617437; examples/sec: 378.1; progress: 61.5%; 
    FastEstimator-Train: step: 12400; meta_loss: 2.286556; inner_loss: 2.2744887; examples/sec: 370.2; progress: 62.0%; 
    FastEstimator-Train: step: 12500; meta_loss: 2.9709299; inner_loss: 3.085735; examples/sec: 368.5; progress: 62.5%; 
    FastEstimator-Train: step: 12600; meta_loss: 3.1649654; inner_loss: 3.5497787; examples/sec: 366.1; progress: 63.0%; 
    FastEstimator-Train: step: 12700; meta_loss: 3.6995928; inner_loss: 3.8092957; examples/sec: 366.2; progress: 63.5%; 
    FastEstimator-Train: step: 12800; meta_loss: 3.3088422; inner_loss: 2.6473522; examples/sec: 365.5; progress: 64.0%; 
    FastEstimator-Train: step: 12900; meta_loss: 3.3592064; inner_loss: 3.2953115; examples/sec: 374.3; progress: 64.5%; 
    FastEstimator-Train: step: 13000; meta_loss: 3.0006561; inner_loss: 3.3691957; examples/sec: 379.8; progress: 65.0%; 
    FastEstimator-Train: step: 13100; meta_loss: 2.774506; inner_loss: 2.8231378; examples/sec: 378.3; progress: 65.5%; 
    FastEstimator-Train: step: 13200; meta_loss: 3.799182; inner_loss: 3.9009297; examples/sec: 380.2; progress: 66.0%; 
    FastEstimator-Train: step: 13300; meta_loss: 2.6867194; inner_loss: 2.9257648; examples/sec: 377.6; progress: 66.5%; 
    FastEstimator-Train: step: 13400; meta_loss: 3.207861; inner_loss: 3.3840263; examples/sec: 378.7; progress: 67.0%; 
    FastEstimator-Train: step: 13500; meta_loss: 3.0611348; inner_loss: 2.9313369; examples/sec: 378.7; progress: 67.5%; 
    FastEstimator-Train: step: 13600; meta_loss: 2.5197492; inner_loss: 2.704917; examples/sec: 380.1; progress: 68.0%; 
    FastEstimator-Train: step: 13700; meta_loss: 3.39369; inner_loss: 3.429712; examples/sec: 376.3; progress: 68.5%; 
    FastEstimator-Train: step: 13800; meta_loss: 3.1631653; inner_loss: 3.4606051; examples/sec: 376.7; progress: 69.0%; 
    FastEstimator-Train: step: 13900; meta_loss: 2.641903; inner_loss: 2.9153838; examples/sec: 381.2; progress: 69.5%; 
    FastEstimator-Train: step: 14000; meta_loss: 3.6415691; inner_loss: 3.44953; examples/sec: 379.4; progress: 70.0%; 
    FastEstimator-Train: step: 14100; meta_loss: 2.3338163; inner_loss: 2.5125897; examples/sec: 378.6; progress: 70.5%; 
    FastEstimator-Train: step: 14200; meta_loss: 4.1154175; inner_loss: 4.1629515; examples/sec: 378.2; progress: 71.0%; 
    FastEstimator-Train: step: 14300; meta_loss: 3.3460715; inner_loss: 3.9577124; examples/sec: 375.8; progress: 71.5%; 
    FastEstimator-Train: step: 14400; meta_loss: 3.3336978; inner_loss: 3.5155385; examples/sec: 378.6; progress: 72.0%; 
    FastEstimator-Train: step: 14500; meta_loss: 3.2897246; inner_loss: 3.2789214; examples/sec: 378.6; progress: 72.5%; 
    FastEstimator-Train: step: 14600; meta_loss: 1.4918921; inner_loss: 1.2954717; examples/sec: 376.8; progress: 73.0%; 
    FastEstimator-Train: step: 14700; meta_loss: 2.948186; inner_loss: 3.0716631; examples/sec: 374.7; progress: 73.5%; 
    FastEstimator-Train: step: 14800; meta_loss: 3.1954143; inner_loss: 3.7012029; examples/sec: 370.7; progress: 74.0%; 
    FastEstimator-Train: step: 14900; meta_loss: 3.194355; inner_loss: 3.0356483; examples/sec: 370.4; progress: 74.5%; 
    FastEstimator-Train: step: 15000; meta_loss: 2.5701962; inner_loss: 2.5250936; examples/sec: 373.0; progress: 75.0%; 
    FastEstimator-Train: step: 15100; meta_loss: 2.7264934; inner_loss: 3.2285776; examples/sec: 380.1; progress: 75.5%; 
    FastEstimator-Train: step: 15200; meta_loss: 3.4389567; inner_loss: 3.64482; examples/sec: 378.9; progress: 76.0%; 
    FastEstimator-Train: step: 15300; meta_loss: 2.5895684; inner_loss: 2.4866898; examples/sec: 378.0; progress: 76.5%; 
    FastEstimator-Train: step: 15400; meta_loss: 3.612536; inner_loss: 4.0838027; examples/sec: 380.2; progress: 77.0%; 
    FastEstimator-Train: step: 15500; meta_loss: 2.9138875; inner_loss: 2.733051; examples/sec: 376.1; progress: 77.5%; 
    FastEstimator-Train: step: 15600; meta_loss: 2.773349; inner_loss: 3.1364257; examples/sec: 380.9; progress: 78.0%; 
    FastEstimator-Train: step: 15700; meta_loss: 2.9843593; inner_loss: 3.1296396; examples/sec: 379.8; progress: 78.5%; 
    FastEstimator-Train: step: 15800; meta_loss: 2.92373; inner_loss: 2.5858984; examples/sec: 378.6; progress: 79.0%; 
    FastEstimator-Train: step: 15900; meta_loss: 2.6386256; inner_loss: 3.0193248; examples/sec: 375.4; progress: 79.5%; 
    FastEstimator-Train: step: 16000; meta_loss: 3.3073392; inner_loss: 3.386622; examples/sec: 371.3; progress: 80.0%; 
    FastEstimator-Train: step: 16100; meta_loss: 3.5852835; inner_loss: 3.8871503; examples/sec: 366.0; progress: 80.5%; 
    FastEstimator-Train: step: 16200; meta_loss: 2.8717449; inner_loss: 3.348681; examples/sec: 367.0; progress: 81.0%; 
    FastEstimator-Train: step: 16300; meta_loss: 2.6117249; inner_loss: 2.6784344; examples/sec: 374.6; progress: 81.5%; 
    FastEstimator-Train: step: 16400; meta_loss: 3.4868617; inner_loss: 3.6717508; examples/sec: 375.6; progress: 82.0%; 
    FastEstimator-Train: step: 16500; meta_loss: 2.9796169; inner_loss: 2.908281; examples/sec: 378.4; progress: 82.5%; 
    FastEstimator-Train: step: 16600; meta_loss: 2.0716202; inner_loss: 2.3175213; examples/sec: 379.3; progress: 83.0%; 
    FastEstimator-Train: step: 16700; meta_loss: 2.3633509; inner_loss: 2.7131226; examples/sec: 377.7; progress: 83.5%; 
    FastEstimator-Train: step: 16800; meta_loss: 2.2930062; inner_loss: 2.955697; examples/sec: 377.4; progress: 84.0%; 
    FastEstimator-Train: step: 16900; meta_loss: 3.357057; inner_loss: 3.2963054; examples/sec: 375.6; progress: 84.5%; 
    FastEstimator-Train: step: 17000; meta_loss: 3.1182098; inner_loss: 2.6884127; examples/sec: 379.6; progress: 85.0%; 
    FastEstimator-Train: step: 17100; meta_loss: 2.9484832; inner_loss: 2.99876; examples/sec: 378.2; progress: 85.5%; 
    FastEstimator-Train: step: 17200; meta_loss: 2.616718; inner_loss: 2.7693486; examples/sec: 377.0; progress: 86.0%; 
    FastEstimator-Train: step: 17300; meta_loss: 2.8318524; inner_loss: 3.3963675; examples/sec: 378.6; progress: 86.5%; 
    FastEstimator-Train: step: 17400; meta_loss: 2.1441848; inner_loss: 2.0441785; examples/sec: 373.8; progress: 87.0%; 
    FastEstimator-Train: step: 17500; meta_loss: 3.2377203; inner_loss: 3.3433783; examples/sec: 375.8; progress: 87.5%; 
    FastEstimator-Train: step: 17600; meta_loss: 2.576676; inner_loss: 2.4001465; examples/sec: 376.8; progress: 88.0%; 
    FastEstimator-Train: step: 17700; meta_loss: 3.3763568; inner_loss: 3.1208477; examples/sec: 378.7; progress: 88.5%; 
    FastEstimator-Train: step: 17800; meta_loss: 2.86822; inner_loss: 2.6269057; examples/sec: 377.6; progress: 89.0%; 
    FastEstimator-Train: step: 17900; meta_loss: 2.7170317; inner_loss: 2.4927347; examples/sec: 374.7; progress: 89.5%; 
    FastEstimator-Train: step: 18000; meta_loss: 2.7359827; inner_loss: 3.024144; examples/sec: 372.7; progress: 90.0%; 
    FastEstimator-Train: step: 18100; meta_loss: 2.3952234; inner_loss: 2.7429423; examples/sec: 369.7; progress: 90.5%; 
    FastEstimator-Train: step: 18200; meta_loss: 3.49225; inner_loss: 3.5674407; examples/sec: 365.1; progress: 91.0%; 
    FastEstimator-Train: step: 18300; meta_loss: 2.5814874; inner_loss: 2.3655543; examples/sec: 367.1; progress: 91.5%; 
    FastEstimator-Train: step: 18400; meta_loss: 3.0865874; inner_loss: 3.3155284; examples/sec: 374.1; progress: 92.0%; 
    FastEstimator-Train: step: 18500; meta_loss: 4.033018; inner_loss: 3.7744093; examples/sec: 379.7; progress: 92.5%; 
    FastEstimator-Train: step: 18600; meta_loss: 2.1386466; inner_loss: 2.064628; examples/sec: 378.2; progress: 93.0%; 
    FastEstimator-Train: step: 18700; meta_loss: 3.6584716; inner_loss: 3.3150997; examples/sec: 379.7; progress: 93.5%; 
    FastEstimator-Train: step: 18800; meta_loss: 3.640406; inner_loss: 4.0982647; examples/sec: 379.8; progress: 94.0%; 
    FastEstimator-Train: step: 18900; meta_loss: 2.7460375; inner_loss: 2.8250656; examples/sec: 375.6; progress: 94.5%; 
    FastEstimator-Train: step: 19000; meta_loss: 2.8167894; inner_loss: 2.9443326; examples/sec: 378.3; progress: 95.0%; 
    FastEstimator-Train: step: 19100; meta_loss: 2.992496; inner_loss: 3.2687924; examples/sec: 374.7; progress: 95.5%; 
    FastEstimator-Train: step: 19200; meta_loss: 2.627812; inner_loss: 2.4452484; examples/sec: 376.7; progress: 96.0%; 
    FastEstimator-Train: step: 19300; meta_loss: 3.3841147; inner_loss: 3.77944; examples/sec: 375.9; progress: 96.5%; 
    FastEstimator-Train: step: 19400; meta_loss: 3.9368477; inner_loss: 4.531379; examples/sec: 377.8; progress: 97.0%; 
    FastEstimator-Train: step: 19500; meta_loss: 2.4636116; inner_loss: 2.4864075; examples/sec: 376.8; progress: 97.5%; 
    FastEstimator-Train: step: 19600; meta_loss: 3.257432; inner_loss: 3.3228183; examples/sec: 377.8; progress: 98.0%; 
    FastEstimator-Train: step: 19700; meta_loss: 2.4407341; inner_loss: 2.7663164; examples/sec: 380.2; progress: 98.5%; 
    FastEstimator-Train: step: 19800; meta_loss: 3.0345795; inner_loss: 3.4228473; examples/sec: 379.1; progress: 99.0%; 
    FastEstimator-Train: step: 19900; meta_loss: 3.062919; inner_loss: 3.087138; examples/sec: 377.5; progress: 99.5%; 
    FastEstimator-Finish: step: 20000; total_time: 1333.5 sec; meta_model_lr: 0.001; 


## Evaluate the Meta Model
Now let's evaluate how good the trained meta model is on the evaluation set that contains only 10 data points of $x$ and $y$.


```python
import matplotlib
import matplotlib.pyplot as plt
```

The orange crosses are available sample data points and the blue curve is the underlying sinusoid curve.
The green curve is a curve fitted by the meta model without any adaptation.


```python
y_sample = sine_test_data["y_meta_train"][:10].astype(np.float32)
x_sample = sine_test_data["x_meta_train"][:10].astype(np.float32)
plt.figure()
x_uniform = np.arange(start=-5, stop=5, step=.0001)
y_uniform = sine_test_data['amp'] * np.sin(x_uniform + sine_test_data['phase'])
plt.plot(x_uniform, y_uniform)
plt.plot(x_sample, y_sample, 'x')
y_pred = meta_model(np.expand_dims(x_uniform, axis=-1))
plt.plot(x_uniform, y_pred)
```

    WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    





    [<matplotlib.lines.Line2D at 0x7fc1b05761d0>]




![png](assets/example/meta_learning/maml_files/maml_19_2.png)


Now we will adapt the meta model to our given sinusoid regression task with only 10 gradient steps.


```python
ds = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_sample,axis=-1), np.expand_dims(y_sample,axis=-1)))
ds = ds.batch(10)
```


```python
def build_task_model(meta_model):
    m = tf.keras.models.clone_model(meta_model)
    m.set_weights(meta_model.get_weights())
    return m
```


```python
task_model = build_task_model(meta_model)
loss_obj = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(2e-2)
for _ in range(10):
    for ex in ds:
        x_tr = ex[0]
        y_tr = ex[1]
        with tf.GradientTape() as t:
            logit = task_model(x_tr, training=True)
            loss = loss_obj(y_tr, logit)
        g = t.gradient(loss, task_model.trainable_variables)
        opt.apply_gradients(zip(g, task_model.trainable_variables))
```

The blue curve and the green curve corresponds to the ground truth and the initial fitted curve from the unadapted model respectively. The red curve is the output of the adapted model. 
We can see that the red curve can approximate the blue curve reasonably well with few gradient update.


```python
plt.figure()
plt.plot(x_uniform, y_uniform)
plt.plot(x_sample, y_sample, 'x')
plt.plot(x_uniform, y_pred)
y_pred_after = task_model(np.expand_dims(x_uniform, axis=-1))
plt.plot(x_uniform, y_pred_after)
```




    [<matplotlib.lines.Line2D at 0x7fc1a866d9b0>]




![png](assets/example/meta_learning/maml_files/maml_25_1.png)

