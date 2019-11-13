# Tutorial 11: Interpretation
___

In this tutorial we introduce the interpretation module, which provides various mechanisms for users to visualize / interpret what a neural network is seeing / focusing on within an input in order to make its decision. Since it will demonstrate bash commands which require certain path information, __this notebook should be run with the tutorial folder as the root directory__.

All interpretation modules may be invoked in three ways:   
    1) From the command line: fastestimator visualize COMMAND   
    2) From a python API: visualize_COMMAND()   
    3) From a __Trace__ which may then be combined with the __TensorBoard__ __Trace__ or any other image IO __Trace__

## Download a sample model for demonstration


```python
import tensorflow as tf
import os

model = tf.keras.applications.InceptionV3(weights='imagenet')  # This will download a .h5 file to ~/.keras/models
os.makedirs('./outputs', exist_ok=True)
model.save('./outputs/inceptionV3.h5')
```

## Interpretation with Bash

We'll start by running the caricature interpretation via the command line. Depending on how you downloaded FastEstimator, you may need to run `python setup.py install` from the parent directory in order to invoke the fastestimator command. For the next few examples we'll be considering an image of a pirate ship: <div>&quot;<a href='https://www.flickr.com/photos/torley/3104607205/' target='_blank'>pirate ship by teepunch Jacobus</a>&quot;&nbsp;(<a rel='license' href='https://creativecommons.org/licenses/by-sa/2.0/' target='_blank'>CC BY-SA 2.0</a>)&nbsp;by&nbsp;<a xmlns:cc='http://creativecommons.org/ns#' rel='cc:attributionURL' property='cc:attributionName' href='https://www.flickr.com/people/torley/' target='_blank'>TORLEY</a></div>


```python
!fastestimator visualize caricature ./outputs/inceptionV3.h5 ./image/pirates.jpg --layers 196 --dictionary ./image/imagenet_class_index.json --save ./outputs
```

    2019-10-23 16:47:50.113010: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    2019-10-23 16:47:50.120515: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300055000 Hz
    2019-10-23 16:47:50.120773: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x697a770 executing computations on platform Host. Devices:
    2019-10-23 16:47:50.120804: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
    2019-10-23 16:47:50.124037: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
    2019-10-23 16:47:50.183820: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.184297: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x57107e0 executing computations on platform CUDA. Devices:
    2019-10-23 16:47:50.184328: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
    2019-10-23 16:47:50.184539: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.184909: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:1e.0
    2019-10-23 16:47:50.185276: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-10-23 16:47:50.186795: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-10-23 16:47:50.188094: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-10-23 16:47:50.188437: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-10-23 16:47:50.190130: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-10-23 16:47:50.191433: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-10-23 16:47:50.195428: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-10-23 16:47:50.195605: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.196106: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.196446: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-10-23 16:47:50.196532: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-10-23 16:47:50.197994: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-10-23 16:47:50.198020: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
    2019-10-23 16:47:50.198037: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
    2019-10-23 16:47:50.198188: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.198623: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.199004: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/device:GPU:0 with 220 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0, compute capability: 3.7)
    2019-10-23 16:47:50.408697: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.409137: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:1e.0
    2019-10-23 16:47:50.409238: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-10-23 16:47:50.409282: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-10-23 16:47:50.409314: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-10-23 16:47:50.409341: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-10-23 16:47:50.409368: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-10-23 16:47:50.409396: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-10-23 16:47:50.409440: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-10-23 16:47:50.409542: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.409957: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.410293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-10-23 16:47:50.410827: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.411178: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
    name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
    pciBusID: 0000:00:1e.0
    2019-10-23 16:47:50.411240: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
    2019-10-23 16:47:50.411280: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-10-23 16:47:50.411308: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
    2019-10-23 16:47:50.411334: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
    2019-10-23 16:47:50.411360: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
    2019-10-23 16:47:50.411386: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
    2019-10-23 16:47:50.411413: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-10-23 16:47:50.411505: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.411934: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.412264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
    2019-10-23 16:47:50.412312: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-10-23 16:47:50.412336: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
    2019-10-23 16:47:50.412354: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
    2019-10-23 16:47:50.412488: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.412900: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2019-10-23 16:47:50.413256: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 220 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:1e.0, compute capability: 3.7)
    2019-10-23 16:47:55.822137: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
    2019-10-23 16:47:58.024002: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 549.62MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.040775: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 208.66MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.051103: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.04GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.068241: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 305.05MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.077520: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
    2019-10-23 16:47:58.745664: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.93GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.748833: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 82.05MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.748878: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 211.75MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.757875: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 216.27MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.760107: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 42.88MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.760145: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 45.32MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
    2019-10-23 16:47:58.849902: W tensorflow/core/kernels/conv_ops.cc:1014] Failed to allocate memory for convolution redzone checking; skipping this check. This is benign and only means that we won't check cudnn for out-of-bounds reads and writes. This message will only be printed once.
    2019-10-23 16:48:08.873435: W tensorflow/core/common_runtime/bfc_allocator.cc:419] Allocator (GPU_0_bfc) ran out of memory trying to allocate 216.8KiB (rounded to 221952).  Current allocation summary follows.
    2019-10-23 16:48:08.873506: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (256): 	Total Chunks: 77, Chunks in use: 76. 19.2KiB allocated for chunks. 19.0KiB in use in bin. 16.8KiB client-requested in use in bin.
    2019-10-23 16:48:08.873526: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (512): 	Total Chunks: 227, Chunks in use: 226. 155.0KiB allocated for chunks. 154.5KiB in use in bin. 143.8KiB client-requested in use in bin.
    2019-10-23 16:48:08.873536: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (1024): 	Total Chunks: 76, Chunks in use: 73. 112.5KiB allocated for chunks. 108.0KiB in use in bin. 106.8KiB client-requested in use in bin.
    2019-10-23 16:48:08.873550: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (2048): 	Total Chunks: 3, Chunks in use: 3. 8.0KiB allocated for chunks. 8.0KiB in use in bin. 6.1KiB client-requested in use in bin.
    2019-10-23 16:48:08.873559: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (4096): 	Total Chunks: 2, Chunks in use: 1. 8.8KiB allocated for chunks. 4.0KiB in use in bin. 3.9KiB client-requested in use in bin.
    2019-10-23 16:48:08.873572: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (8192): 	Total Chunks: 1, Chunks in use: 0. 13.5KiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
    2019-10-23 16:48:08.873580: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (16384): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
    2019-10-23 16:48:08.873594: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (32768): 	Total Chunks: 7, Chunks in use: 6. 306.5KiB allocated for chunks. 270.5KiB in use in bin. 212.0KiB client-requested in use in bin.
    2019-10-23 16:48:08.873605: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (65536): 	Total Chunks: 14, Chunks in use: 13. 1.13MiB allocated for chunks. 1.06MiB in use in bin. 979.1KiB client-requested in use in bin.
    2019-10-23 16:48:08.873619: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (131072): 	Total Chunks: 116, Chunks in use: 116. 21.99MiB allocated for chunks. 21.99MiB in use in bin. 21.91MiB client-requested in use in bin.
    2019-10-23 16:48:08.873630: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (262144): 	Total Chunks: 77, Chunks in use: 77. 27.95MiB allocated for chunks. 27.95MiB in use in bin. 26.84MiB client-requested in use in bin.
    2019-10-23 16:48:08.873643: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (524288): 	Total Chunks: 45, Chunks in use: 45. 34.70MiB allocated for chunks. 34.70MiB in use in bin. 31.99MiB client-requested in use in bin.
    2019-10-23 16:48:08.873653: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (1048576): 	Total Chunks: 23, Chunks in use: 23. 33.94MiB allocated for chunks. 33.94MiB in use in bin. 30.32MiB client-requested in use in bin.
    2019-10-23 16:48:08.873667: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (2097152): 	Total Chunks: 17, Chunks in use: 17. 48.46MiB allocated for chunks. 48.46MiB in use in bin. 42.41MiB client-requested in use in bin.
    2019-10-23 16:48:08.873677: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (4194304): 	Total Chunks: 8, Chunks in use: 8. 41.95MiB allocated for chunks. 41.95MiB in use in bin. 37.65MiB client-requested in use in bin.
    2019-10-23 16:48:08.873691: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (8388608): 	Total Chunks: 1, Chunks in use: 1. 9.72MiB allocated for chunks. 9.72MiB in use in bin. 7.81MiB client-requested in use in bin.
    2019-10-23 16:48:08.873711: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (16777216): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
    2019-10-23 16:48:08.873728: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (33554432): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
    2019-10-23 16:48:08.873745: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (67108864): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
    2019-10-23 16:48:08.873760: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (134217728): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
    2019-10-23 16:48:08.873767: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (268435456): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
    2019-10-23 16:48:08.873780: I tensorflow/core/common_runtime/bfc_allocator.cc:885] Bin for 216.8KiB was 128.0KiB, Chunk State: 
    2019-10-23 16:48:08.873787: I tensorflow/core/common_runtime/bfc_allocator.cc:898] Next region of size 231145472
    2019-10-23 16:48:08.873801: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60000 next 1 of size 1280
    2019-10-23 16:48:08.873807: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60500 next 2 of size 256
    2019-10-23 16:48:08.873818: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60600 next 5 of size 256
    2019-10-23 16:48:08.873825: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60700 next 4 of size 256
    2019-10-23 16:48:08.873831: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60800 next 7 of size 256
    2019-10-23 16:48:08.873842: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60900 next 8 of size 256
    2019-10-23 16:48:08.873847: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60a00 next 9 of size 256
    2019-10-23 16:48:08.873855: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60b00 next 11 of size 256
    2019-10-23 16:48:08.873870: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60c00 next 13 of size 256
    2019-10-23 16:48:08.873876: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60d00 next 14 of size 256
    2019-10-23 16:48:08.873881: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60e00 next 16 of size 256
    2019-10-23 16:48:08.873893: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b60f00 next 18 of size 256
    2019-10-23 16:48:08.873899: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61000 next 19 of size 256
    2019-10-23 16:48:08.873910: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61100 next 20 of size 256
    2019-10-23 16:48:08.873918: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61200 next 21 of size 512
    2019-10-23 16:48:08.873928: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61400 next 22 of size 512
    2019-10-23 16:48:08.873934: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61600 next 23 of size 512
    2019-10-23 16:48:08.873946: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61800 next 25 of size 512
    2019-10-23 16:48:08.873953: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61a00 next 26 of size 768
    2019-10-23 16:48:08.873964: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b61d00 next 28 of size 768
    2019-10-23 16:48:08.873970: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b62000 next 3 of size 1024
    2019-10-23 16:48:08.873981: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b62400 next 494 of size 512
    2019-10-23 16:48:08.873986: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b62600 next 495 of size 768
    2019-10-23 16:48:08.873998: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b62900 next 114 of size 1536
    2019-10-23 16:48:08.874004: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b62f00 next 30 of size 1536
    2019-10-23 16:48:08.874014: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b63500 next 32 of size 256
    2019-10-23 16:48:08.874021: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b63600 next 34 of size 256
    2019-10-23 16:48:08.874032: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b63700 next 35 of size 256
    2019-10-23 16:48:08.874038: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b63800 next 36 of size 256
    2019-10-23 16:48:08.874048: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b63900 next 215 of size 768
    2019-10-23 16:48:08.874054: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b63c00 next 216 of size 768
    2019-10-23 16:48:08.874065: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b63f00 next 218 of size 768
    2019-10-23 16:48:08.874071: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b64200 next 219 of size 768
    2019-10-23 16:48:08.874081: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b64500 next 220 of size 768
    2019-10-23 16:48:08.874087: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b64800 next 221 of size 768
    2019-10-23 16:48:08.874094: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b64b00 next 222 of size 768
    2019-10-23 16:48:08.874106: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b64e00 next 224 of size 768
    2019-10-23 16:48:08.874123: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b65100 next 226 of size 768
    2019-10-23 16:48:08.874138: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b65400 next 227 of size 768
    2019-10-23 16:48:08.874155: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b65700 next 230 of size 768
    2019-10-23 16:48:08.874171: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b65a00 next 232 of size 768
    2019-10-23 16:48:08.874188: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b65d00 next 233 of size 768
    2019-10-23 16:48:08.874204: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b66000 next 234 of size 768
    2019-10-23 16:48:08.874210: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b66300 next 235 of size 768
    2019-10-23 16:48:08.874221: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b66600 next 236 of size 768
    2019-10-23 16:48:08.874226: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b66900 next 237 of size 768
    2019-10-23 16:48:08.874238: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b66c00 next 238 of size 768
    2019-10-23 16:48:08.874243: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b66f00 next 241 of size 768
    2019-10-23 16:48:08.874254: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b67200 next 242 of size 768
    2019-10-23 16:48:08.874260: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b67500 next 243 of size 768
    2019-10-23 16:48:08.874270: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b67800 next 244 of size 768
    2019-10-23 16:48:08.874276: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b67b00 next 245 of size 768
    2019-10-23 16:48:08.874286: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b67e00 next 246 of size 768
    2019-10-23 16:48:08.874292: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b68100 next 247 of size 768
    2019-10-23 16:48:08.874302: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b68400 next 248 of size 768
    2019-10-23 16:48:08.874308: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b68700 next 252 of size 768
    2019-10-23 16:48:08.874318: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b68a00 next 253 of size 768
    2019-10-23 16:48:08.874324: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b68d00 next 254 of size 768
    2019-10-23 16:48:08.874334: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b69000 next 255 of size 768
    2019-10-23 16:48:08.874340: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b69300 next 256 of size 768
    2019-10-23 16:48:08.874350: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b69600 next 257 of size 768
    2019-10-23 16:48:08.874356: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b69900 next 258 of size 768
    2019-10-23 16:48:08.874363: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b69c00 next 259 of size 768
    2019-10-23 16:48:08.874370: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b69f00 next 260 of size 768
    2019-10-23 16:48:08.874375: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6a200 next 261 of size 768
    2019-10-23 16:48:08.874387: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6a500 next 262 of size 768
    2019-10-23 16:48:08.874392: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6a800 next 263 of size 768
    2019-10-23 16:48:08.874397: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6ab00 next 264 of size 768
    2019-10-23 16:48:08.874409: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6ae00 next 265 of size 768
    2019-10-23 16:48:08.874414: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6b100 next 266 of size 768
    2019-10-23 16:48:08.874419: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6b400 next 267 of size 768
    2019-10-23 16:48:08.874426: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6b700 next 268 of size 768
    2019-10-23 16:48:08.874433: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6ba00 next 269 of size 768
    2019-10-23 16:48:08.874439: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6bd00 next 270 of size 768
    2019-10-23 16:48:08.874451: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6c000 next 271 of size 768
    2019-10-23 16:48:08.874457: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6c300 next 272 of size 768
    2019-10-23 16:48:08.874467: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6c600 next 274 of size 768
    2019-10-23 16:48:08.874472: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6c900 next 276 of size 768
    2019-10-23 16:48:08.874484: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6cc00 next 277 of size 768
    2019-10-23 16:48:08.874489: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6cf00 next 24 of size 768
    2019-10-23 16:48:08.874500: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b6d200 next 12 of size 32768
    2019-10-23 16:48:08.874506: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b75200 next 10 of size 36864
    2019-10-23 16:48:08.874516: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203b7e200 next 37 of size 36864
    2019-10-23 16:48:08.874522: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87200 next 38 of size 256
    2019-10-23 16:48:08.874532: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87300 next 40 of size 256
    2019-10-23 16:48:08.874537: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87400 next 42 of size 256
    2019-10-23 16:48:08.874548: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87500 next 43 of size 256
    2019-10-23 16:48:08.874554: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87600 next 44 of size 512
    2019-10-23 16:48:08.874564: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87800 next 45 of size 512
    2019-10-23 16:48:08.874569: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87a00 next 46 of size 512
    2019-10-23 16:48:08.874580: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87c00 next 47 of size 512
    2019-10-23 16:48:08.874592: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b87e00 next 147 of size 512
    2019-10-23 16:48:08.874602: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b88000 next 149 of size 512
    2019-10-23 16:48:08.874608: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b88200 next 150 of size 512
    2019-10-23 16:48:08.874618: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b88400 next 151 of size 512
    2019-10-23 16:48:08.874628: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b88600 next 154 of size 1536
    2019-10-23 16:48:08.874639: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b88c00 next 157 of size 1536
    2019-10-23 16:48:08.874645: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b89200 next 153 of size 768
    2019-10-23 16:48:08.874651: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b89500 next 159 of size 2304
    2019-10-23 16:48:08.874665: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b89e00 next 160 of size 512
    2019-10-23 16:48:08.874671: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8a000 next 161 of size 512
    2019-10-23 16:48:08.874676: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8a200 next 162 of size 512
    2019-10-23 16:48:08.874683: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8a400 next 163 of size 512
    2019-10-23 16:48:08.874690: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8a600 next 164 of size 512
    2019-10-23 16:48:08.874696: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8a800 next 166 of size 512
    2019-10-23 16:48:08.874703: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8aa00 next 168 of size 512
    2019-10-23 16:48:08.874709: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8ac00 next 169 of size 512
    2019-10-23 16:48:08.874722: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8ae00 next 171 of size 512
    2019-10-23 16:48:08.874727: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8b000 next 173 of size 512
    2019-10-23 16:48:08.874737: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8b200 next 174 of size 512
    2019-10-23 16:48:08.874743: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8b400 next 175 of size 512
    2019-10-23 16:48:08.874753: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8b600 next 177 of size 512
    2019-10-23 16:48:08.874759: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8b800 next 180 of size 512
    2019-10-23 16:48:08.874769: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8ba00 next 181 of size 512
    2019-10-23 16:48:08.874775: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8bc00 next 182 of size 512
    2019-10-23 16:48:08.874785: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8be00 next 183 of size 512
    2019-10-23 16:48:08.874790: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8c000 next 184 of size 512
    2019-10-23 16:48:08.874800: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8c200 next 185 of size 512
    2019-10-23 16:48:08.874806: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8c400 next 186 of size 512
    2019-10-23 16:48:08.874816: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8c600 next 189 of size 512
    2019-10-23 16:48:08.874822: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8c800 next 190 of size 512
    2019-10-23 16:48:08.874828: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8ca00 next 191 of size 512
    2019-10-23 16:48:08.874839: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8cc00 next 192 of size 512
    2019-10-23 16:48:08.874845: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8ce00 next 193 of size 512
    2019-10-23 16:48:08.874855: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8d000 next 194 of size 512
    2019-10-23 16:48:08.874861: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8d200 next 195 of size 512
    2019-10-23 16:48:08.874872: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203b8d400 next 196 of size 512
    2019-10-23 16:48:08.874877: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8d600 next 199 of size 768
    2019-10-23 16:48:08.874883: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8d900 next 202 of size 768
    2019-10-23 16:48:08.874894: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8dc00 next 204 of size 768
    2019-10-23 16:48:08.874899: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8df00 next 205 of size 768
    2019-10-23 16:48:08.874907: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8e200 next 206 of size 768
    2019-10-23 16:48:08.874912: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8e500 next 207 of size 768
    2019-10-23 16:48:08.874919: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8e800 next 208 of size 768
    2019-10-23 16:48:08.874931: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8eb00 next 209 of size 768
    2019-10-23 16:48:08.874947: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8ee00 next 210 of size 768
    2019-10-23 16:48:08.874958: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8f100 next 211 of size 768
    2019-10-23 16:48:08.874970: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8f400 next 212 of size 768
    2019-10-23 16:48:08.874976: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8f700 next 213 of size 768
    2019-10-23 16:48:08.874986: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8fa00 next 214 of size 768
    2019-10-23 16:48:08.874991: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b8fd00 next 17 of size 1280
    2019-10-23 16:48:08.875002: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203b90200 next 15 of size 73728
    2019-10-23 16:48:08.875008: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba2200 next 240 of size 1792
    2019-10-23 16:48:08.875020: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba2900 next 91 of size 1536
    2019-10-23 16:48:08.875025: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba2f00 next 420 of size 1536
    2019-10-23 16:48:08.875036: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba3500 next 155 of size 1792
    2019-10-23 16:48:08.875042: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba3c00 next 490 of size 4096
    2019-10-23 16:48:08.875052: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203ba4c00 next 53 of size 13824
    2019-10-23 16:48:08.875057: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8200 next 55 of size 256
    2019-10-23 16:48:08.875068: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8300 next 56 of size 256
    2019-10-23 16:48:08.875074: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8400 next 57 of size 256
    2019-10-23 16:48:08.875084: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8500 next 58 of size 256
    2019-10-23 16:48:08.875090: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8600 next 59 of size 256
    2019-10-23 16:48:08.875100: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8700 next 60 of size 256
    2019-10-23 16:48:08.875106: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8800 next 61 of size 256
    2019-10-23 16:48:08.875116: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8900 next 62 of size 256
    2019-10-23 16:48:08.875127: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8a00 next 63 of size 512
    2019-10-23 16:48:08.875134: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8c00 next 64 of size 512
    2019-10-23 16:48:08.875140: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba8e00 next 65 of size 512
    2019-10-23 16:48:08.875150: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9000 next 66 of size 512
    2019-10-23 16:48:08.875156: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9200 next 67 of size 256
    2019-10-23 16:48:08.875168: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9300 next 68 of size 256
    2019-10-23 16:48:08.875177: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9400 next 69 of size 256
    2019-10-23 16:48:08.875187: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9500 next 70 of size 256
    2019-10-23 16:48:08.875198: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9600 next 71 of size 256
    2019-10-23 16:48:08.875206: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9700 next 72 of size 256
    2019-10-23 16:48:08.875216: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9800 next 73 of size 256
    2019-10-23 16:48:08.875226: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9900 next 75 of size 256
    2019-10-23 16:48:08.875234: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9a00 next 77 of size 256
    2019-10-23 16:48:08.875253: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9b00 next 79 of size 256
    2019-10-23 16:48:08.875263: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9c00 next 80 of size 256
    2019-10-23 16:48:08.875273: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9d00 next 81 of size 256
    2019-10-23 16:48:08.875283: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ba9e00 next 82 of size 512
    2019-10-23 16:48:08.875292: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baa000 next 83 of size 512
    2019-10-23 16:48:08.875305: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baa200 next 84 of size 512
    2019-10-23 16:48:08.875316: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baa400 next 85 of size 512
    2019-10-23 16:48:08.875331: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baa600 next 88 of size 256
    2019-10-23 16:48:08.875341: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baa700 next 92 of size 256
    2019-10-23 16:48:08.875357: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baa800 next 93 of size 256
    2019-10-23 16:48:08.875368: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baa900 next 94 of size 256
    2019-10-23 16:48:08.875383: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baaa00 next 95 of size 256
    2019-10-23 16:48:08.875393: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baab00 next 96 of size 256
    2019-10-23 16:48:08.875404: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baac00 next 97 of size 256
    2019-10-23 16:48:08.875414: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baad00 next 98 of size 256
    2019-10-23 16:48:08.875430: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baae00 next 99 of size 512
    2019-10-23 16:48:08.875439: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bab000 next 100 of size 512
    2019-10-23 16:48:08.875449: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bab200 next 101 of size 512
    2019-10-23 16:48:08.875459: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bab400 next 102 of size 512
    2019-10-23 16:48:08.875473: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bab600 next 103 of size 256
    2019-10-23 16:48:08.875484: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bab700 next 104 of size 256
    2019-10-23 16:48:08.875496: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bab800 next 105 of size 256
    2019-10-23 16:48:08.875501: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bab900 next 106 of size 256
    2019-10-23 16:48:08.875512: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baba00 next 108 of size 256
    2019-10-23 16:48:08.875517: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203babb00 next 110 of size 256
    2019-10-23 16:48:08.875528: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203babc00 next 111 of size 256
    2019-10-23 16:48:08.875533: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203babd00 next 112 of size 256
    2019-10-23 16:48:08.875544: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203babe00 next 115 of size 256
    2019-10-23 16:48:08.875549: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203babf00 next 116 of size 256
    2019-10-23 16:48:08.875556: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bac000 next 117 of size 256
    2019-10-23 16:48:08.875569: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bac100 next 118 of size 256
    2019-10-23 16:48:08.875574: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bac200 next 119 of size 512
    2019-10-23 16:48:08.875584: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bac400 next 120 of size 512
    2019-10-23 16:48:08.875590: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bac600 next 121 of size 512
    2019-10-23 16:48:08.875600: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bac800 next 122 of size 512
    2019-10-23 16:48:08.875606: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203baca00 next 126 of size 256
    2019-10-23 16:48:08.875617: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bacb00 next 127 of size 256
    2019-10-23 16:48:08.875622: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bacc00 next 128 of size 256
    2019-10-23 16:48:08.875629: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bacd00 next 129 of size 256
    2019-10-23 16:48:08.875635: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bace00 next 130 of size 256
    2019-10-23 16:48:08.875645: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bacf00 next 131 of size 256
    2019-10-23 16:48:08.875651: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bad000 next 132 of size 256
    2019-10-23 16:48:08.875663: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bad100 next 133 of size 256
    2019-10-23 16:48:08.875673: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bad200 next 134 of size 512
    2019-10-23 16:48:08.875683: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bad400 next 135 of size 512
    2019-10-23 16:48:08.875698: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bad600 next 136 of size 512
    2019-10-23 16:48:08.875709: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bad800 next 137 of size 512
    2019-10-23 16:48:08.875721: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bada00 next 138 of size 256
    2019-10-23 16:48:08.875727: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203badb00 next 139 of size 256
    2019-10-23 16:48:08.875737: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203badc00 next 140 of size 256
    2019-10-23 16:48:08.875743: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203badd00 next 141 of size 256
    2019-10-23 16:48:08.875753: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bade00 next 143 of size 256
    2019-10-23 16:48:08.875765: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203badf00 next 144 of size 256
    2019-10-23 16:48:08.875781: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bae000 next 145 of size 256
    2019-10-23 16:48:08.875821: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203bae100 next 33 of size 256
    2019-10-23 16:48:08.875833: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bae200 next 31 of size 49152
    2019-10-23 16:48:08.875844: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bba200 next 113 of size 276480
    2019-10-23 16:48:08.875856: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bfda00 next 481 of size 1280
    2019-10-23 16:48:08.875869: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bfdf00 next 482 of size 1280
    2019-10-23 16:48:08.875885: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bfe400 next 483 of size 1280
    2019-10-23 16:48:08.875900: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203bfe900 next 484 of size 1280
    2019-10-23 16:48:08.875906: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bfee00 next 485 of size 768
    2019-10-23 16:48:08.875916: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bff100 next 486 of size 768
    2019-10-23 16:48:08.875922: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203bff400 next 487 of size 768
    2019-10-23 16:48:08.875933: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203bff700 next 489 of size 4864
    2019-10-23 16:48:08.875939: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c00a00 next 491 of size 3584
    2019-10-23 16:48:08.875953: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c01800 next 89 of size 59904
    2019-10-23 16:48:08.875959: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c10200 next 41 of size 90112
    2019-10-23 16:48:08.875969: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c26200 next 54 of size 65536
    2019-10-23 16:48:08.875980: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c36200 next 74 of size 65536
    2019-10-23 16:48:08.875992: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c46200 next 39 of size 90112
    2019-10-23 16:48:08.876007: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5c200 next 280 of size 768
    2019-10-23 16:48:08.876022: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5c500 next 282 of size 768
    2019-10-23 16:48:08.876033: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5c800 next 283 of size 768
    2019-10-23 16:48:08.876044: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5cb00 next 284 of size 768
    2019-10-23 16:48:08.876054: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5ce00 next 285 of size 768
    2019-10-23 16:48:08.876065: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5d100 next 286 of size 768
    2019-10-23 16:48:08.876071: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5d400 next 287 of size 768
    2019-10-23 16:48:08.876081: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5d700 next 290 of size 768
    2019-10-23 16:48:08.876093: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5da00 next 291 of size 768
    2019-10-23 16:48:08.876108: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5dd00 next 292 of size 768
    2019-10-23 16:48:08.876123: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5e000 next 293 of size 768
    2019-10-23 16:48:08.876134: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5e300 next 294 of size 768
    2019-10-23 16:48:08.876140: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5e600 next 295 of size 768
    2019-10-23 16:48:08.876151: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5e900 next 296 of size 768
    2019-10-23 16:48:08.876156: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5ec00 next 297 of size 768
    2019-10-23 16:48:08.876168: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5ef00 next 301 of size 768
    2019-10-23 16:48:08.876173: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5f200 next 302 of size 768
    2019-10-23 16:48:08.876186: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5f500 next 303 of size 768
    2019-10-23 16:48:08.876192: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5f800 next 304 of size 768
    2019-10-23 16:48:08.876208: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5fb00 next 305 of size 768
    2019-10-23 16:48:08.876213: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c5fe00 next 306 of size 768
    2019-10-23 16:48:08.876224: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c60100 next 307 of size 768
    2019-10-23 16:48:08.876229: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c60400 next 308 of size 768
    2019-10-23 16:48:08.876239: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c60700 next 309 of size 768
    2019-10-23 16:48:08.876254: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c60a00 next 310 of size 768
    2019-10-23 16:48:08.876260: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c60d00 next 311 of size 768
    2019-10-23 16:48:08.876270: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c61000 next 312 of size 768
    2019-10-23 16:48:08.876280: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c61300 next 313 of size 768
    2019-10-23 16:48:08.876286: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c61600 next 314 of size 768
    2019-10-23 16:48:08.876296: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c61900 next 315 of size 768
    2019-10-23 16:48:08.876308: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c61c00 next 316 of size 768
    2019-10-23 16:48:08.876324: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c61f00 next 317 of size 768
    2019-10-23 16:48:08.876338: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c62200 next 318 of size 768
    2019-10-23 16:48:08.876349: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c62500 next 319 of size 768
    2019-10-23 16:48:08.876360: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c62800 next 320 of size 768
    2019-10-23 16:48:08.876371: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c62b00 next 321 of size 768
    2019-10-23 16:48:08.876381: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c62e00 next 323 of size 768
    2019-10-23 16:48:08.876393: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c63100 next 325 of size 768
    2019-10-23 16:48:08.876399: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c63400 next 326 of size 768
    2019-10-23 16:48:08.878704: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c63700 next 329 of size 768
    2019-10-23 16:48:08.878712: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c63a00 next 331 of size 768
    2019-10-23 16:48:08.878719: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c63d00 next 332 of size 768
    2019-10-23 16:48:08.878726: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c64000 next 333 of size 768
    2019-10-23 16:48:08.878732: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c64300 next 334 of size 768
    2019-10-23 16:48:08.878739: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c64600 next 335 of size 768
    2019-10-23 16:48:08.878746: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c64900 next 336 of size 768
    2019-10-23 16:48:08.878758: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c64c00 next 337 of size 768
    2019-10-23 16:48:08.878764: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c64f00 next 340 of size 768
    2019-10-23 16:48:08.878770: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c65200 next 341 of size 768
    2019-10-23 16:48:08.878777: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c65500 next 342 of size 768
    2019-10-23 16:48:08.878788: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c65800 next 343 of size 768
    2019-10-23 16:48:08.878794: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c65b00 next 344 of size 768
    2019-10-23 16:48:08.878800: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c65e00 next 345 of size 768
    2019-10-23 16:48:08.878811: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c66100 next 346 of size 768
    2019-10-23 16:48:08.878817: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c66400 next 347 of size 768
    2019-10-23 16:48:08.878827: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c66700 next 351 of size 768
    2019-10-23 16:48:08.878832: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c66a00 next 352 of size 768
    2019-10-23 16:48:08.878843: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c66d00 next 353 of size 768
    2019-10-23 16:48:08.878848: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c67000 next 354 of size 768
    2019-10-23 16:48:08.878859: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c67300 next 355 of size 768
    2019-10-23 16:48:08.878865: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c67600 next 356 of size 768
    2019-10-23 16:48:08.878875: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c67900 next 357 of size 768
    2019-10-23 16:48:08.878881: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c67c00 next 358 of size 768
    2019-10-23 16:48:08.878891: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c67f00 next 50 of size 768
    2019-10-23 16:48:08.878897: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c68200 next 48 of size 49152
    2019-10-23 16:48:08.878903: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c74200 next 359 of size 768
    2019-10-23 16:48:08.878915: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c74500 next 360 of size 768
    2019-10-23 16:48:08.878931: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c74800 next 361 of size 768
    2019-10-23 16:48:08.878947: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c74b00 next 362 of size 768
    2019-10-23 16:48:08.878962: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c74e00 next 363 of size 768
    2019-10-23 16:48:08.878974: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c75100 next 364 of size 768
    2019-10-23 16:48:08.878986: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c75400 next 365 of size 768
    2019-10-23 16:48:08.878992: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c75700 next 367 of size 768
    2019-10-23 16:48:08.879002: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c75a00 next 368 of size 768
    2019-10-23 16:48:08.879007: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c75d00 next 369 of size 768
    2019-10-23 16:48:08.879019: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c76000 next 370 of size 768
    2019-10-23 16:48:08.879024: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c76300 next 371 of size 768
    2019-10-23 16:48:08.879031: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c76600 next 373 of size 768
    2019-10-23 16:48:08.879041: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c76900 next 375 of size 768
    2019-10-23 16:48:08.879046: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c76c00 next 376 of size 768
    2019-10-23 16:48:08.879056: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c76f00 next 377 of size 768
    2019-10-23 16:48:08.879062: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c77200 next 379 of size 768
    2019-10-23 16:48:08.879072: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c77500 next 380 of size 768
    2019-10-23 16:48:08.879078: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c77800 next 381 of size 768
    2019-10-23 16:48:08.879088: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c77b00 next 382 of size 768
    2019-10-23 16:48:08.879093: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c77e00 next 383 of size 768
    2019-10-23 16:48:08.879103: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c78100 next 384 of size 768
    2019-10-23 16:48:08.879109: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c78400 next 385 of size 768
    2019-10-23 16:48:08.879119: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c78700 next 388 of size 1280
    2019-10-23 16:48:08.879125: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c78c00 next 391 of size 1280
    2019-10-23 16:48:08.879135: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c79100 next 392 of size 1280
    2019-10-23 16:48:08.879141: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c79600 next 393 of size 1280
    2019-10-23 16:48:08.879151: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c79b00 next 394 of size 768
    2019-10-23 16:48:08.879157: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c79e00 next 395 of size 768
    2019-10-23 16:48:08.879163: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7a100 next 396 of size 768
    2019-10-23 16:48:08.879169: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7a400 next 397 of size 768
    2019-10-23 16:48:08.879179: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7a700 next 398 of size 1792
    2019-10-23 16:48:08.879184: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7ae00 next 400 of size 1792
    2019-10-23 16:48:08.879190: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7b500 next 402 of size 1792
    2019-10-23 16:48:08.879197: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7bc00 next 403 of size 1792
    2019-10-23 16:48:08.879211: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7c300 next 404 of size 1536
    2019-10-23 16:48:08.879222: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7c900 next 407 of size 1536
    2019-10-23 16:48:08.879231: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7cf00 next 409 of size 1536
    2019-10-23 16:48:08.879241: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7d500 next 410 of size 1536
    2019-10-23 16:48:08.879251: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7db00 next 411 of size 1536
    2019-10-23 16:48:08.879260: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7e100 next 412 of size 1536
    2019-10-23 16:48:08.879276: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7e700 next 413 of size 1536
    2019-10-23 16:48:08.879287: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7ed00 next 414 of size 1536
    2019-10-23 16:48:08.879297: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7f300 next 419 of size 1536
    2019-10-23 16:48:08.879306: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c7f900 next 78 of size 2304
    2019-10-23 16:48:08.879317: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203c80200 next 158 of size 393216
    2019-10-23 16:48:08.879327: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203ce0200 next 421 of size 424960
    2019-10-23 16:48:08.879342: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d47e00 next 422 of size 1536
    2019-10-23 16:48:08.879352: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d48400 next 423 of size 1536
    2019-10-23 16:48:08.879367: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d48a00 next 424 of size 1536
    2019-10-23 16:48:08.879377: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d49000 next 425 of size 1536
    2019-10-23 16:48:08.879392: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d49600 next 426 of size 1536
    2019-10-23 16:48:08.879401: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d49c00 next 427 of size 1536
    2019-10-23 16:48:08.879416: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4a200 next 428 of size 1536
    2019-10-23 16:48:08.879426: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4a800 next 429 of size 1536
    2019-10-23 16:48:08.879436: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4ae00 next 430 of size 1536
    2019-10-23 16:48:08.879445: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4b400 next 431 of size 1536
    2019-10-23 16:48:08.879454: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4ba00 next 432 of size 1536
    2019-10-23 16:48:08.879464: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4c000 next 433 of size 1536
    2019-10-23 16:48:08.879473: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4c600 next 435 of size 1280
    2019-10-23 16:48:08.879484: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4cb00 next 436 of size 1280
    2019-10-23 16:48:08.879493: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4d000 next 437 of size 1280
    2019-10-23 16:48:08.879507: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4d500 next 438 of size 1280
    2019-10-23 16:48:08.879518: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4da00 next 439 of size 768
    2019-10-23 16:48:08.879532: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4dd00 next 440 of size 768
    2019-10-23 16:48:08.879542: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4e000 next 441 of size 768
    2019-10-23 16:48:08.879557: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4e300 next 442 of size 768
    2019-10-23 16:48:08.879567: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4e600 next 444 of size 1792
    2019-10-23 16:48:08.879581: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4ed00 next 446 of size 1792
    2019-10-23 16:48:08.879592: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d4f400 next 447 of size 1792
    2019-10-23 16:48:08.879605: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203d4fb00 next 448 of size 1792
    2019-10-23 16:48:08.879616: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d50200 next 450 of size 1536
    2019-10-23 16:48:08.879630: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d50800 next 452 of size 1536
    2019-10-23 16:48:08.879641: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d50e00 next 454 of size 1536
    2019-10-23 16:48:08.879656: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d51400 next 455 of size 1536
    2019-10-23 16:48:08.879666: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d51a00 next 456 of size 1536
    2019-10-23 16:48:08.879676: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d52000 next 457 of size 1536
    2019-10-23 16:48:08.879686: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d52600 next 458 of size 1536
    2019-10-23 16:48:08.879702: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d52c00 next 459 of size 1536
    2019-10-23 16:48:08.879711: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d53200 next 463 of size 1536
    2019-10-23 16:48:08.879721: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d53800 next 465 of size 1536
    2019-10-23 16:48:08.879730: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d53e00 next 467 of size 1536
    2019-10-23 16:48:08.879739: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d54400 next 468 of size 1536
    2019-10-23 16:48:08.879753: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d54a00 next 469 of size 1536
    2019-10-23 16:48:08.879764: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d55000 next 470 of size 1536
    2019-10-23 16:48:08.879774: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d55600 next 471 of size 1536
    2019-10-23 16:48:08.879797: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d55c00 next 472 of size 1536
    2019-10-23 16:48:08.879812: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d56200 next 473 of size 1536
    2019-10-23 16:48:08.879822: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d56800 next 474 of size 1536
    2019-10-23 16:48:08.879836: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d56e00 next 475 of size 1536
    2019-10-23 16:48:08.879846: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d57400 next 476 of size 1536
    2019-10-23 16:48:08.879856: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d57a00 next 477 of size 1536
    2019-10-23 16:48:08.879866: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d58000 next 478 of size 1536
    2019-10-23 16:48:08.879874: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d58600 next 479 of size 1536
    2019-10-23 16:48:08.879888: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203d58c00 next 109 of size 1536
    2019-10-23 16:48:08.879899: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d59200 next 107 of size 73728
    2019-10-23 16:48:08.879909: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d6b200 next 52 of size 94208
    2019-10-23 16:48:08.879919: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203d82200 next 49 of size 307200
    2019-10-23 16:48:08.879928: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203dcd200 next 125 of size 73728
    2019-10-23 16:48:08.879943: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x1203ddf200 next 142 of size 73728
    2019-10-23 16:48:08.879953: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203df1200 next 493 of size 73728
    2019-10-23 16:48:08.879967: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203e03200 next 6 of size 49152
    2019-10-23 16:48:08.879978: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203e0f200 next 51 of size 393216
    2019-10-23 16:48:08.879987: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203e6f200 next 87 of size 73728
    2019-10-23 16:48:08.879998: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203e81200 next 86 of size 233472
    2019-10-23 16:48:08.880008: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203eba200 next 90 of size 331776
    2019-10-23 16:48:08.880023: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203f0b200 next 27 of size 589824
    2019-10-23 16:48:08.880033: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1203f9b200 next 124 of size 602112
    2019-10-23 16:48:08.880048: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120402e200 next 148 of size 221184
    2019-10-23 16:48:08.880058: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1204064200 next 146 of size 221184
    2019-10-23 16:48:08.880072: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120409a200 next 156 of size 331776
    2019-10-23 16:48:08.880082: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12040eb200 next 152 of size 331776
    2019-10-23 16:48:08.880098: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120413c200 next 178 of size 393216
    2019-10-23 16:48:08.880109: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120419c200 next 167 of size 393216
    2019-10-23 16:48:08.880123: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12041fc200 next 165 of size 393216
    2019-10-23 16:48:08.880133: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120425c200 next 172 of size 458752
    2019-10-23 16:48:08.880149: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12042cc200 next 170 of size 458752
    2019-10-23 16:48:08.880158: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120433c200 next 179 of size 458752
    2019-10-23 16:48:08.880172: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12043ac200 next 176 of size 458752
    2019-10-23 16:48:08.880182: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120441c200 next 187 of size 458752
    2019-10-23 16:48:08.880197: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120448c200 next 188 of size 458752
    2019-10-23 16:48:08.880207: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12044fc200 next 203 of size 589824
    2019-10-23 16:48:08.880217: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120458c200 next 200 of size 589824
    2019-10-23 16:48:08.880227: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120461c200 next 198 of size 589824
    2019-10-23 16:48:08.880236: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12046ac200 next 123 of size 331776
    2019-10-23 16:48:08.880246: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12046fd200 next 278 of size 2293760
    2019-10-23 16:48:08.880256: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120492d200 next 225 of size 5566464
    2019-10-23 16:48:08.880272: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1204e7c200 next 223 of size 716800
    2019-10-23 16:48:08.880282: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1204f2b200 next 231 of size 716800
    2019-10-23 16:48:08.880298: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1204fda200 next 229 of size 716800
    2019-10-23 16:48:08.880308: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205089200 next 239 of size 716800
    2019-10-23 16:48:08.880318: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205138200 next 228 of size 716800
    2019-10-23 16:48:08.880328: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12051e7200 next 251 of size 860160
    2019-10-23 16:48:08.880338: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12052b9200 next 249 of size 860160
    2019-10-23 16:48:08.880347: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120538b200 next 250 of size 860160
    2019-10-23 16:48:08.880361: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120545d200 next 275 of size 1433600
    2019-10-23 16:48:08.880372: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12055bb200 next 273 of size 716800
    2019-10-23 16:48:08.880386: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120566a200 next 281 of size 716800
    2019-10-23 16:48:08.880396: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205719200 next 279 of size 716800
    2019-10-23 16:48:08.880411: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12057c8200 next 288 of size 716800
    2019-10-23 16:48:08.880421: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205877200 next 289 of size 716800
    2019-10-23 16:48:08.880435: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205926200 next 300 of size 860160
    2019-10-23 16:48:08.880445: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12059f8200 next 298 of size 860160
    2019-10-23 16:48:08.880459: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205aca200 next 299 of size 860160
    2019-10-23 16:48:08.880470: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205b9c200 next 327 of size 589824
    2019-10-23 16:48:08.880480: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205c2c200 next 366 of size 589824
    2019-10-23 16:48:08.880490: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205cbc200 next 324 of size 884736
    2019-10-23 16:48:08.880499: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205d94200 next 322 of size 1032192
    2019-10-23 16:48:08.880517: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205e90200 next 330 of size 1032192
    2019-10-23 16:48:08.880527: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1205f8c200 next 328 of size 1032192
    2019-10-23 16:48:08.880543: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206088200 next 338 of size 1032192
    2019-10-23 16:48:08.880553: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206184200 next 339 of size 1032192
    2019-10-23 16:48:08.880563: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206280200 next 350 of size 1032192
    2019-10-23 16:48:08.880573: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120637c200 next 348 of size 589824
    2019-10-23 16:48:08.880589: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120640c200 next 349 of size 1474560
    2019-10-23 16:48:08.880600: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206574200 next 374 of size 2064384
    2019-10-23 16:48:08.880615: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120676c200 next 372 of size 1032192
    2019-10-23 16:48:08.880621: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206868200 next 378 of size 1032192
    2019-10-23 16:48:08.880633: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206964200 next 390 of size 1327104
    2019-10-23 16:48:08.880639: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206aa8200 next 386 of size 1327104
    2019-10-23 16:48:08.880650: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206bec200 next 201 of size 688128
    2019-10-23 16:48:08.880655: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206c94200 next 389 of size 1081344
    2019-10-23 16:48:08.880666: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206d9c200 next 387 of size 2211840
    2019-10-23 16:48:08.880672: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1206fb8200 next 434 of size 983040
    2019-10-23 16:48:08.880679: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12070a8200 next 76 of size 458752
    2019-10-23 16:48:08.880685: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1207118200 next 406 of size 524288
    2019-10-23 16:48:08.880695: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1207198200 next 401 of size 2621440
    2019-10-23 16:48:08.880701: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1207418200 next 399 of size 2293760
    2019-10-23 16:48:08.880707: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1207648200 next 415 of size 1769472
    2019-10-23 16:48:08.880714: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12077f8200 next 416 of size 1769472
    2019-10-23 16:48:08.880727: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12079a8200 next 417 of size 1769472
    2019-10-23 16:48:08.880736: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1207b58200 next 460 of size 3407872
    2019-10-23 16:48:08.880747: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1207e98200 next 461 of size 1769472
    2019-10-23 16:48:08.880757: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1208048200 next 408 of size 1900544
    2019-10-23 16:48:08.880766: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1208218200 next 405 of size 6193152
    2019-10-23 16:48:08.880782: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1208800200 next 217 of size 860160
    2019-10-23 16:48:08.880794: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12088d2200 next 462 of size 909312
    2019-10-23 16:48:08.880809: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12089b0200 next 445 of size 1900544
    2019-10-23 16:48:08.880820: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1208b80200 next 443 of size 3670016
    2019-10-23 16:48:08.880835: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1208f00200 next 449 of size 3145728
    2019-10-23 16:48:08.880845: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1209200200 next 466 of size 5242880
    2019-10-23 16:48:08.880860: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1209700200 next 480 of size 4194304
    2019-10-23 16:48:08.880870: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1209b00200 next 453 of size 2949120
    2019-10-23 16:48:08.880885: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1209dd0200 next 418 of size 2621440
    2019-10-23 16:48:08.880901: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120a050200 next 509 of size 470528
    2019-10-23 16:48:08.880919: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120a0c3000 next 510 of size 235264
    2019-10-23 16:48:08.880930: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120a0fc700 next 500 of size 367104
    2019-10-23 16:48:08.880943: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120a156100 next 501 of size 1072896
    2019-10-23 16:48:08.880949: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120a25c000 next 451 of size 1425920
    2019-10-23 16:48:08.880956: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120a3b8200 next 197 of size 6193152
    2019-10-23 16:48:08.880964: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120a9a0200 next 492 of size 10190848
    2019-10-23 16:48:08.880970: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120b358200 next 498 of size 2841856
    2019-10-23 16:48:08.880983: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120b60df00 next 502 of size 2841856
    2019-10-23 16:48:08.880989: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120b8c3c00 next 503 of size 2766080
    2019-10-23 16:48:08.880994: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120bb67100 next 499 of size 2766080
    2019-10-23 16:48:08.881001: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120be0a600 next 496 of size 2766080
    2019-10-23 16:48:08.881009: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120c0adb00 next 497 of size 5531904
    2019-10-23 16:48:08.881016: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120c5f4400 next 464 of size 5531904
    2019-10-23 16:48:08.881023: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120cb3ad00 next 29 of size 5531904
    2019-10-23 16:48:08.881030: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120d081600 next 488 of size 1705472
    2019-10-23 16:48:08.881040: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120d221c00 next 504 of size 1705472
    2019-10-23 16:48:08.881046: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120d3c2200 next 505 of size 1705472
    2019-10-23 16:48:08.881057: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120d562800 next 506 of size 3871488
    2019-10-23 16:48:08.881063: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120d913b00 next 507 of size 3871488
    2019-10-23 16:48:08.881076: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120dcc4e00 next 508 of size 3871488
    2019-10-23 16:48:08.881082: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e076100 next 511 of size 470528
    2019-10-23 16:48:08.881092: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e0e8f00 next 512 of size 470528
    2019-10-23 16:48:08.881098: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e15bd00 next 513 of size 313600
    2019-10-23 16:48:08.881108: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e1a8600 next 514 of size 313600
    2019-10-23 16:48:08.881114: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e1f4f00 next 515 of size 156928
    2019-10-23 16:48:08.881124: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e21b400 next 516 of size 470528
    2019-10-23 16:48:08.881130: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e28e200 next 517 of size 313600
    2019-10-23 16:48:08.881140: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e2dab00 next 518 of size 313600
    2019-10-23 16:48:08.881146: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e327400 next 519 of size 313600
    2019-10-23 16:48:08.881159: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e373d00 next 520 of size 313600
    2019-10-23 16:48:08.881165: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e3c0600 next 521 of size 470528
    2019-10-23 16:48:08.881177: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e433400 next 522 of size 156928
    2019-10-23 16:48:08.881182: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e459900 next 523 of size 1254400
    2019-10-23 16:48:08.881194: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e58bd00 next 524 of size 313600
    2019-10-23 16:48:08.881199: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e5d8600 next 525 of size 313600
    2019-10-23 16:48:08.881206: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e624f00 next 526 of size 313600
    2019-10-23 16:48:08.881212: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e671800 next 527 of size 470528
    2019-10-23 16:48:08.881218: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e6e4600 next 528 of size 235264
    2019-10-23 16:48:08.881225: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e71dd00 next 529 of size 470528
    2019-10-23 16:48:08.881231: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e790b00 next 530 of size 235264
    2019-10-23 16:48:08.881248: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e7ca200 next 531 of size 1254400
    2019-10-23 16:48:08.881254: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e8fc600 next 532 of size 470528
    2019-10-23 16:48:08.881259: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e96f400 next 533 of size 235264
    2019-10-23 16:48:08.881266: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e9a8b00 next 534 of size 313600
    2019-10-23 16:48:08.881273: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120e9f5400 next 535 of size 470528
    2019-10-23 16:48:08.881279: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ea68200 next 536 of size 313600
    2019-10-23 16:48:08.881286: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120eab4b00 next 537 of size 313600
    2019-10-23 16:48:08.881293: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120eb01400 next 538 of size 313600
    2019-10-23 16:48:08.881305: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120eb4dd00 next 539 of size 470528
    2019-10-23 16:48:08.881311: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ebc0b00 next 540 of size 313600
    2019-10-23 16:48:08.881318: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ec0d400 next 541 of size 313600
    2019-10-23 16:48:08.881328: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ec59d00 next 542 of size 313600
    2019-10-23 16:48:08.881333: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120eca6600 next 543 of size 313600
    2019-10-23 16:48:08.881344: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ecf2f00 next 544 of size 470528
    2019-10-23 16:48:08.881349: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ed65d00 next 545 of size 313600
    2019-10-23 16:48:08.881360: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120edb2600 next 546 of size 1411328
    2019-10-23 16:48:08.881366: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ef0af00 next 547 of size 313600
    2019-10-23 16:48:08.881376: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ef57800 next 548 of size 313600
    2019-10-23 16:48:08.881382: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120efa4100 next 549 of size 313600
    2019-10-23 16:48:08.881394: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120eff0a00 next 550 of size 470528
    2019-10-23 16:48:08.881400: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f063800 next 551 of size 235264
    2019-10-23 16:48:08.881411: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f09cf00 next 552 of size 470528
    2019-10-23 16:48:08.881416: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f10fd00 next 553 of size 235264
    2019-10-23 16:48:08.881426: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f149400 next 554 of size 1411328
    2019-10-23 16:48:08.881432: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f2a1d00 next 555 of size 470528
    2019-10-23 16:48:08.881442: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f314b00 next 556 of size 235264
    2019-10-23 16:48:08.881448: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f34e200 next 557 of size 313600
    2019-10-23 16:48:08.881459: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f39ab00 next 558 of size 470528
    2019-10-23 16:48:08.881464: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f40d900 next 559 of size 313600
    2019-10-23 16:48:08.883544: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f45a200 next 560 of size 313600
    2019-10-23 16:48:08.883577: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f4a6b00 next 561 of size 313600
    2019-10-23 16:48:08.883590: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f4f3400 next 562 of size 470528
    2019-10-23 16:48:08.883600: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f566200 next 563 of size 313600
    2019-10-23 16:48:08.883614: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f5b2b00 next 564 of size 313600
    2019-10-23 16:48:08.883625: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f5ff400 next 565 of size 313600
    2019-10-23 16:48:08.883640: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f64bd00 next 566 of size 313600
    2019-10-23 16:48:08.883651: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f698600 next 567 of size 470528
    2019-10-23 16:48:08.883666: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f70b400 next 568 of size 313600
    2019-10-23 16:48:08.883676: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f757d00 next 569 of size 1411328
    2019-10-23 16:48:08.883691: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f8b0600 next 570 of size 313600
    2019-10-23 16:48:08.883701: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f8fcf00 next 571 of size 313600
    2019-10-23 16:48:08.883716: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f949800 next 572 of size 313600
    2019-10-23 16:48:08.883726: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120f996100 next 573 of size 470528
    2019-10-23 16:48:08.883741: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fa08f00 next 574 of size 470528
    2019-10-23 16:48:08.883751: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fa7bd00 next 575 of size 470528
    2019-10-23 16:48:08.883768: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120faeeb00 next 576 of size 111104
    2019-10-23 16:48:08.883778: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fb09d00 next 577 of size 443904
    2019-10-23 16:48:08.883821: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fb76300 next 578 of size 111104
    2019-10-23 16:48:08.883833: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fb91500 next 579 of size 443904
    2019-10-23 16:48:08.883843: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fbfdb00 next 580 of size 443904
    2019-10-23 16:48:08.883861: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fc6a100 next 581 of size 111104
    2019-10-23 16:48:08.883871: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fc85300 next 582 of size 333056
    2019-10-23 16:48:08.883888: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fcd6800 next 583 of size 887808
    2019-10-23 16:48:08.883900: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fdaf400 next 584 of size 147968
    2019-10-23 16:48:08.883909: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fdd3600 next 585 of size 147968
    2019-10-23 16:48:08.883924: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fdf7800 next 586 of size 147968
    2019-10-23 16:48:08.883940: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fe1ba00 next 587 of size 147968
    2019-10-23 16:48:08.883955: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fe3fc00 next 588 of size 147968
    2019-10-23 16:48:08.883971: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fe63e00 next 589 of size 147968
    2019-10-23 16:48:08.883982: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fe88000 next 590 of size 147968
    2019-10-23 16:48:08.883990: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120feac200 next 591 of size 147968
    2019-10-23 16:48:08.883996: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fed0400 next 592 of size 147968
    2019-10-23 16:48:08.884007: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fef4600 next 593 of size 147968
    2019-10-23 16:48:08.884013: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ff18800 next 594 of size 147968
    2019-10-23 16:48:08.884018: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ff3ca00 next 595 of size 147968
    2019-10-23 16:48:08.884025: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ff60c00 next 596 of size 147968
    2019-10-23 16:48:08.884030: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ff84e00 next 597 of size 147968
    2019-10-23 16:48:08.884037: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ffa9000 next 598 of size 147968
    2019-10-23 16:48:08.884043: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120ffcd200 next 599 of size 147968
    2019-10-23 16:48:08.884057: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x120fff1400 next 600 of size 887808
    2019-10-23 16:48:08.884066: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12100ca000 next 601 of size 147968
    2019-10-23 16:48:08.884081: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12100ee200 next 602 of size 147968
    2019-10-23 16:48:08.884088: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210112400 next 603 of size 221952
    2019-10-23 16:48:08.884095: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210148700 next 604 of size 221952
    2019-10-23 16:48:08.884102: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121017ea00 next 605 of size 221952
    2019-10-23 16:48:08.884108: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12101b4d00 next 606 of size 221952
    2019-10-23 16:48:08.884118: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12101eb000 next 607 of size 221952
    2019-10-23 16:48:08.884124: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210221300 next 608 of size 221952
    2019-10-23 16:48:08.884131: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210257600 next 609 of size 221952
    2019-10-23 16:48:08.884136: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121028d900 next 610 of size 221952
    2019-10-23 16:48:08.884147: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12102c3c00 next 611 of size 221952
    2019-10-23 16:48:08.884152: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12102f9f00 next 612 of size 221952
    2019-10-23 16:48:08.884159: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210330200 next 613 of size 221952
    2019-10-23 16:48:08.884165: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210366500 next 614 of size 221952
    2019-10-23 16:48:08.884176: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121039c800 next 615 of size 887808
    2019-10-23 16:48:08.884182: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210475400 next 616 of size 185088
    2019-10-23 16:48:08.884189: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12104a2700 next 617 of size 185088
    2019-10-23 16:48:08.884194: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12104cfa00 next 618 of size 185088
    2019-10-23 16:48:08.884204: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12104fcd00 next 619 of size 185088
    2019-10-23 16:48:08.884210: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121052a000 next 620 of size 185088
    2019-10-23 16:48:08.884217: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210557300 next 621 of size 185088
    2019-10-23 16:48:08.884223: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210584600 next 622 of size 185088
    2019-10-23 16:48:08.884234: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12105b1900 next 623 of size 185088
    2019-10-23 16:48:08.884240: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12105dec00 next 624 of size 185088
    2019-10-23 16:48:08.884246: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121060bf00 next 625 of size 185088
    2019-10-23 16:48:08.884252: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210639200 next 626 of size 185088
    2019-10-23 16:48:08.884263: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210666500 next 627 of size 185088
    2019-10-23 16:48:08.884268: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210693800 next 628 of size 185088
    2019-10-23 16:48:08.884274: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12106c0b00 next 629 of size 185088
    2019-10-23 16:48:08.884280: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12106ede00 next 630 of size 185088
    2019-10-23 16:48:08.884286: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121071b100 next 631 of size 185088
    2019-10-23 16:48:08.884293: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210748400 next 632 of size 887808
    2019-10-23 16:48:08.884298: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210821000 next 633 of size 185088
    2019-10-23 16:48:08.884312: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121084e300 next 634 of size 185088
    2019-10-23 16:48:08.884322: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121087b600 next 635 of size 221952
    2019-10-23 16:48:08.884337: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12108b1900 next 636 of size 221952
    2019-10-23 16:48:08.884348: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12108e7c00 next 637 of size 221952
    2019-10-23 16:48:08.884358: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121091df00 next 638 of size 221952
    2019-10-23 16:48:08.884368: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210954200 next 639 of size 221952
    2019-10-23 16:48:08.884378: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121098a500 next 640 of size 221952
    2019-10-23 16:48:08.884387: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12109c0800 next 641 of size 221952
    2019-10-23 16:48:08.884402: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12109f6b00 next 642 of size 221952
    2019-10-23 16:48:08.884411: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210a2ce00 next 643 of size 221952
    2019-10-23 16:48:08.884426: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210a63100 next 644 of size 221952
    2019-10-23 16:48:08.884442: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210a99400 next 645 of size 221952
    2019-10-23 16:48:08.884457: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210acf700 next 646 of size 221952
    2019-10-23 16:48:08.884473: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210b05a00 next 647 of size 887808
    2019-10-23 16:48:08.884484: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210bde600 next 648 of size 185088
    2019-10-23 16:48:08.884492: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210c0b900 next 649 of size 185088
    2019-10-23 16:48:08.884498: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210c38c00 next 650 of size 185088
    2019-10-23 16:48:08.884508: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210c65f00 next 651 of size 185088
    2019-10-23 16:48:08.884514: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210c93200 next 652 of size 185088
    2019-10-23 16:48:08.884519: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210cc0500 next 653 of size 185088
    2019-10-23 16:48:08.884526: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210ced800 next 654 of size 185088
    2019-10-23 16:48:08.884531: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210d1ab00 next 655 of size 185088
    2019-10-23 16:48:08.884538: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210d47e00 next 656 of size 185088
    2019-10-23 16:48:08.884543: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210d75100 next 657 of size 185088
    2019-10-23 16:48:08.884550: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210da2400 next 658 of size 185088
    2019-10-23 16:48:08.884556: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210dcf700 next 659 of size 185088
    2019-10-23 16:48:08.884569: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210dfca00 next 660 of size 185088
    2019-10-23 16:48:08.884581: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210e29d00 next 661 of size 185088
    2019-10-23 16:48:08.884592: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210e57000 next 662 of size 185088
    2019-10-23 16:48:08.884601: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210e84300 next 663 of size 185088
    2019-10-23 16:48:08.884613: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210eb1600 next 664 of size 887808
    2019-10-23 16:48:08.884629: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210f8a200 next 665 of size 185088
    2019-10-23 16:48:08.884645: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210fb7500 next 666 of size 185088
    2019-10-23 16:48:08.884660: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1210fe4800 next 667 of size 221952
    2019-10-23 16:48:08.884672: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121101ab00 next 668 of size 221952
    2019-10-23 16:48:08.884680: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211050e00 next 669 of size 221952
    2019-10-23 16:48:08.884685: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211087100 next 670 of size 221952
    2019-10-23 16:48:08.884696: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12110bd400 next 671 of size 221952
    2019-10-23 16:48:08.884702: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12110f3700 next 672 of size 221952
    2019-10-23 16:48:08.884709: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211129a00 next 673 of size 221952
    2019-10-23 16:48:08.884714: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121115fd00 next 674 of size 221952
    2019-10-23 16:48:08.884724: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211196000 next 675 of size 221952
    2019-10-23 16:48:08.884730: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12111cc300 next 676 of size 221952
    2019-10-23 16:48:08.884736: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211202600 next 677 of size 221952
    2019-10-23 16:48:08.884748: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211238900 next 678 of size 221952
    2019-10-23 16:48:08.884763: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121126ec00 next 679 of size 887808
    2019-10-23 16:48:08.884779: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211347800 next 680 of size 221952
    2019-10-23 16:48:08.884795: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121137db00 next 681 of size 221952
    2019-10-23 16:48:08.884811: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12113b3e00 next 682 of size 221952
    2019-10-23 16:48:08.884827: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12113ea100 next 683 of size 221952
    2019-10-23 16:48:08.884843: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211420400 next 684 of size 221952
    2019-10-23 16:48:08.884858: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211456700 next 685 of size 221952
    2019-10-23 16:48:08.884869: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121148ca00 next 686 of size 221952
    2019-10-23 16:48:08.884878: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12114c2d00 next 687 of size 221952
    2019-10-23 16:48:08.884883: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12114f9000 next 688 of size 221952
    2019-10-23 16:48:08.884894: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121152f300 next 689 of size 221952
    2019-10-23 16:48:08.884900: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211565600 next 690 of size 221952
    2019-10-23 16:48:08.884906: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121159b900 next 691 of size 221952
    2019-10-23 16:48:08.884912: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x12115d1c00 next 692 of size 221952
    2019-10-23 16:48:08.884922: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x1211607f00 next 693 of size 221952
    2019-10-23 16:48:08.884928: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x121163e200 next 18446744073709551615 of size 1646080
    2019-10-23 16:48:08.884935: I tensorflow/core/common_runtime/bfc_allocator.cc:914]      Summary of in-use Chunks by size: 
    2019-10-23 16:48:08.884950: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 76 Chunks of size 256 totalling 19.0KiB
    2019-10-23 16:48:08.884961: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 60 Chunks of size 512 totalling 30.0KiB
    2019-10-23 16:48:08.884969: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 166 Chunks of size 768 totalling 124.5KiB
    2019-10-23 16:48:08.884980: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1024 totalling 1.0KiB
    2019-10-23 16:48:08.884986: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 13 Chunks of size 1280 totalling 16.2KiB
    2019-10-23 16:48:08.884997: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 50 Chunks of size 1536 totalling 75.0KiB
    2019-10-23 16:48:08.885003: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 9 Chunks of size 1792 totalling 15.8KiB
    2019-10-23 16:48:08.885010: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 2304 totalling 4.5KiB
    2019-10-23 16:48:08.885016: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 3584 totalling 3.5KiB
    2019-10-23 16:48:08.885023: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 4096 totalling 4.0KiB
    2019-10-23 16:48:08.885029: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 32768 totalling 32.0KiB
    2019-10-23 16:48:08.885036: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 36864 totalling 36.0KiB
    2019-10-23 16:48:08.885042: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 49152 totalling 144.0KiB
    2019-10-23 16:48:08.885049: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 59904 totalling 58.5KiB
    2019-10-23 16:48:08.885055: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 65536 totalling 128.0KiB
    2019-10-23 16:48:08.885062: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 5 Chunks of size 73728 totalling 360.0KiB
    2019-10-23 16:48:08.885068: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 90112 totalling 176.0KiB
    2019-10-23 16:48:08.885075: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 94208 totalling 92.0KiB
    2019-10-23 16:48:08.885081: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 111104 totalling 325.5KiB
    2019-10-23 16:48:08.885094: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 18 Chunks of size 147968 totalling 2.54MiB
    2019-10-23 16:48:08.885100: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 156928 totalling 306.5KiB
    2019-10-23 16:48:08.885107: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 36 Chunks of size 185088 totalling 6.35MiB
    2019-10-23 16:48:08.885115: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 221184 totalling 432.0KiB
    2019-10-23 16:48:08.885128: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 50 Chunks of size 221952 totalling 10.58MiB
    2019-10-23 16:48:08.885145: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 233472 totalling 228.0KiB
    2019-10-23 16:48:08.885157: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 7 Chunks of size 235264 totalling 1.57MiB
    2019-10-23 16:48:08.885165: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 276480 totalling 270.0KiB
    2019-10-23 16:48:08.885172: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 307200 totalling 300.0KiB
    2019-10-23 16:48:08.885178: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 33 Chunks of size 313600 totalling 9.87MiB
    2019-10-23 16:48:08.885190: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 4 Chunks of size 331776 totalling 1.27MiB
    2019-10-23 16:48:08.885196: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 333056 totalling 325.2KiB
    2019-10-23 16:48:08.885204: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 367104 totalling 358.5KiB
    2019-10-23 16:48:08.885211: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 5 Chunks of size 393216 totalling 1.88MiB
    2019-10-23 16:48:08.885217: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 424960 totalling 415.0KiB
    2019-10-23 16:48:08.885230: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 443904 totalling 1.27MiB
    2019-10-23 16:48:08.885236: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 7 Chunks of size 458752 totalling 3.06MiB
    2019-10-23 16:48:08.885243: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 20 Chunks of size 470528 totalling 8.97MiB
    2019-10-23 16:48:08.885258: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 524288 totalling 512.0KiB
    2019-10-23 16:48:08.885270: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 7 Chunks of size 589824 totalling 3.94MiB
    2019-10-23 16:48:08.885280: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 602112 totalling 588.0KiB
    2019-10-23 16:48:08.885298: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 688128 totalling 672.0KiB
    2019-10-23 16:48:08.885309: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 10 Chunks of size 716800 totalling 6.84MiB
    2019-10-23 16:48:08.885320: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 7 Chunks of size 860160 totalling 5.74MiB
    2019-10-23 16:48:08.885331: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 884736 totalling 864.0KiB
    2019-10-23 16:48:08.885342: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 7 Chunks of size 887808 totalling 5.93MiB
    2019-10-23 16:48:08.885353: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 909312 totalling 888.0KiB
    2019-10-23 16:48:08.885363: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 983040 totalling 960.0KiB
    2019-10-23 16:48:08.885379: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 8 Chunks of size 1032192 totalling 7.88MiB
    2019-10-23 16:48:08.885389: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1072896 totalling 1.02MiB
    2019-10-23 16:48:08.885405: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1081344 totalling 1.03MiB
    2019-10-23 16:48:08.885421: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 1254400 totalling 2.39MiB
    2019-10-23 16:48:08.885433: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 1327104 totalling 2.53MiB
    2019-10-23 16:48:08.885448: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 1411328 totalling 4.04MiB
    2019-10-23 16:48:08.885460: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1425920 totalling 1.36MiB
    2019-10-23 16:48:08.885469: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1433600 totalling 1.37MiB
    2019-10-23 16:48:08.885484: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1474560 totalling 1.41MiB
    2019-10-23 16:48:08.885501: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1646080 totalling 1.57MiB
    2019-10-23 16:48:08.885517: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 1705472 totalling 4.88MiB
    2019-10-23 16:48:08.885528: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 4 Chunks of size 1769472 totalling 6.75MiB
    2019-10-23 16:48:08.885544: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 1900544 totalling 3.62MiB
    2019-10-23 16:48:08.885555: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 2064384 totalling 1.97MiB
    2019-10-23 16:48:08.885566: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 2211840 totalling 2.11MiB
    2019-10-23 16:48:08.885577: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 2293760 totalling 4.38MiB
    2019-10-23 16:48:08.885587: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 2621440 totalling 5.00MiB
    2019-10-23 16:48:08.885596: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 2766080 totalling 7.91MiB
    2019-10-23 16:48:08.885616: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 2841856 totalling 5.42MiB
    2019-10-23 16:48:08.885628: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 2949120 totalling 2.81MiB
    2019-10-23 16:48:08.885648: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 3145728 totalling 3.00MiB
    2019-10-23 16:48:08.885658: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 3407872 totalling 3.25MiB
    2019-10-23 16:48:08.885674: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 3670016 totalling 3.50MiB
    2019-10-23 16:48:08.885692: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 3871488 totalling 11.08MiB
    2019-10-23 16:48:08.885704: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 4194304 totalling 4.00MiB
    2019-10-23 16:48:08.885715: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 5242880 totalling 5.00MiB
    2019-10-23 16:48:08.885724: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 3 Chunks of size 5531904 totalling 15.83MiB
    2019-10-23 16:48:08.885731: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 5566464 totalling 5.31MiB
    2019-10-23 16:48:08.885737: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 2 Chunks of size 6193152 totalling 11.81MiB
    2019-10-23 16:48:08.885756: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 10190848 totalling 9.72MiB
    2019-10-23 16:48:08.885763: I tensorflow/core/common_runtime/bfc_allocator.cc:921] Sum Total of in-use chunks: 220.31MiB
    2019-10-23 16:48:08.885769: I tensorflow/core/common_runtime/bfc_allocator.cc:923] total_region_allocated_bytes_: 231145472 memory_limit_: 231145472 available bytes: 0 curr_region_allocation_bytes_: 462290944
    2019-10-23 16:48:08.885781: I tensorflow/core/common_runtime/bfc_allocator.cc:929] Stats: 
    Limit:                   231145472
    InUse:                   231010816
    MaxInUse:                231010816
    NumAllocs:                    3351
    MaxAllocSize:             71747584
    
    2019-10-23 16:48:08.885831: W tensorflow/core/common_runtime/bfc_allocator.cc:424] ****************************************************************************************************
    2019-10-23 16:48:08.887351: W tensorflow/core/framework/op_kernel.cc:1622] OP_REQUIRES failed at conv_ops.cc:947 : Resource exhausted: OOM when allocating tensor with shape[1,192,17,17] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
    Traceback (most recent call last):
      File "/usr/local/bin/fastestimator", line 7, in <module>
        exec(compile(f.read(), __file__, 'exec'))
      File "/tf/fastestimator/bin/fastestimator", line 35, in <module>
        run()
      File "/tf/fastestimator/bin/fastestimator", line 31, in run
        args.func(vars(args), unknown)
      File "/tf/fastestimator/fastestimator/cli/visualize.py", line 275, in caricature
        sigmoid=not args['hard_clip'])
      File "/tf/fastestimator/fastestimator/cli/cli_util.py", line 160, in load_and_caricature
        sigmoid=sigmoid)
      File "/tf/fastestimator/fastestimator/xai/caricature.py", line 252, in visualize_caricature
        sigmoid=sigmoid)
      File "/tf/fastestimator/fastestimator/xai/caricature.py", line 169, in plot_caricature
        predictions = np.asarray(model(model_input))
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine/base_layer.py", line 851, in __call__
        outputs = self.call(cast_inputs, *args, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine/network.py", line 697, in call
        return self._run_internal_graph(inputs, training=training, mask=mask)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine/network.py", line 842, in _run_internal_graph
        output_tensors = layer(computed_tensors, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/engine/base_layer.py", line 851, in __call__
        outputs = self.call(cast_inputs, *args, **kwargs)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/layers/convolutional.py", line 197, in call
        outputs = self._convolution_op(inputs, self.kernel)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_ops.py", line 1134, in __call__
        return self.conv_op(inp, filter)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_ops.py", line 639, in __call__
        return self.call(inp, filter)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_ops.py", line 238, in __call__
        name=self.name)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/nn_ops.py", line 2010, in conv2d
        name=name)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/gen_nn_ops.py", line 1031, in conv2d
        data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/gen_nn_ops.py", line 1130, in conv2d_eager_fallback
        ctx=_ctx, name=name)
      File "/usr/local/lib/python3.6/dist-packages/tensorflow_core/python/eager/execute.py", line 67, in quick_execute
        six.raise_from(core._status_to_exception(e.code, message), None)
      File "<string>", line 3, in raise_from
    tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[1,192,17,17] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:Conv2D]


Now lets load the image generated from the bash command back into memory for visualization:


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
from fastestimator.xai import show_image

%matplotlib inline

caricature = plt.imread('./outputs/caricatures.png')
mpl.rcParams['figure.dpi']=300
show_image(caricature)
plt.show()
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-3-67497de601a3> in <module>
          1 import matplotlib as mpl
          2 import matplotlib.pyplot as plt
    ----> 3 from fastestimator.xai import show_image
          4 
          5 get_ipython().run_line_magic('matplotlib', 'inline')


    /tf/fastestimator/fastestimator/__init__.py in <module>
         15 import tensorflow as tf
         16 
    ---> 17 from fastestimator.estimator import Estimator
         18 from fastestimator.network import Network, build
         19 from fastestimator.pipeline import Pipeline


    /tf/fastestimator/fastestimator/estimator.py in <module>
         23 from fastestimator.schedule.epoch_scheduler import Scheduler
         24 from fastestimator.summary import Summary
    ---> 25 from fastestimator.trace import Logger, ModelSaver, MonitorLoss, Trace, TrainInfo
         26 from fastestimator.util.util import get_num_devices, per_replica_to_global
         27 


    /tf/fastestimator/fastestimator/trace/__init__.py in <module>
         15 from fastestimator.trace.trace import Trace, TrainInfo, MonitorLoss  # isort:skip
         16 from fastestimator.trace.adapt import EarlyStopping, LRController, TerminateOnNaN
    ---> 17 from fastestimator.trace.io import Caricature, CSVLogger, GradCam, Logger, ModelSaver, Saliency, SlackNotification, \
         18     TensorBoard, UMap, VisLogger
         19 from fastestimator.trace.metric import Accuracy, ConfusionMatrix, Dice, F1Score, Precision, Recall


    /tf/fastestimator/fastestimator/trace/io/__init__.py in <module>
         19 from fastestimator.trace.io.model_saver import ModelSaver
         20 from fastestimator.trace.io.saliency import Saliency
    ---> 21 from fastestimator.trace.io.slackio import SlackNotification
         22 from fastestimator.trace.io.tensorboard import TensorBoard
         23 from fastestimator.trace.io.umap import UMap


    /tf/fastestimator/fastestimator/trace/io/slackio.py in <module>
         17 import types
         18 
    ---> 19 import nest_asyncio
         20 import slack
         21 from slack.errors import SlackApiError


    ModuleNotFoundError: No module named 'nest_asyncio'


A human who was trying to identify a pirate ship would likely focus on 2 key components: whether they are looking at a ship, and whether that ship is flying a pirate flag. From the caricature we can see that the stern of the ship is strongly emphasized, and that there is a large but ambiguously structured area for the sails. Interestingly, the pirate flag is completely absent from the caricature -- perhaps indicating the network is not interested in the flag. Our next interpretation module will investigate this more closesly. 

## Interpretation with Python API

Now lets run a saliency analysis (a different interpretation method), this time via the FastEstimator python API. We'll start by loading the pirate image into memory and sizing it appropriately


```python
from fastestimator.util.util import load_dict, load_image

input_type = model.input.dtype
input_shape = model.input.shape
n_channels = 0 if len(input_shape) == 3 else input_shape[3]
input_height = input_shape[1]
input_width = input_shape[2]
inputs = [load_image('./image/pirates.jpg', channels=n_channels)]
tf_image = tf.stack([
    tf.image.resize_with_pad(tf.convert_to_tensor(im, dtype=input_type),
                             input_height,
                             input_width,
                             method='lanczos3') for im in inputs
])
pirates = tf.clip_by_value(tf_image, -1, 1)
dic = load_dict('./image/imagenet_class_index.json')
baseline = tf.zeros_like(pirates) + 0.5
```

Now lets run the FE saliency api:


```python
from fastestimator.xai import visualize_saliency

visualize_saliency(model, pirates, baseline_input=baseline, decode_dictionary=dic, save_path=None)
```

Here we can see that, as in the caricature analysis, there seems to be strong emphasis on the stern of the ship, with some focus on the sails, but none on the pirate flag. It seems likely that the neural network has not learned that pirates are regular boats with a flag modifier, but rather that it has correlated a certain stereotypical ship design with pirate ships. 

## Interpretation with Traces

We now move on to a discussion of interpretation with Traces. For this example we will switch to the CIFAR10 dataset and see how UMAPs can be used to visualize what a network 'knows' about different classes.


```python
import tensorflow as tf
from tensorflow.python.keras import layers, Sequential, Model

from fastestimator import Pipeline, Network, build, Estimator
from fastestimator.architecture import LeNet
from fastestimator.op.tensorop import Minmax, ModelOp, SparseCategoricalCrossentropy
from tensorflow.python.keras import layers, Sequential
from fastestimator.trace import UMap, ConfusionMatrix, VisLogger


(x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
data = {"train": {"x": x_train, "y": y_train}, "eval": {"x": x_eval, "y": y_eval}}
num_classes = 10
class_dictionary = {
    0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

pipeline = Pipeline(batch_size=32, data=data, ops=Minmax(inputs="x", outputs="x"))

model = build(model_def=lambda: LeNet(input_shape=x_train.shape[1:], classes=num_classes),
              model_name="LeNet",
              optimizer="adam",
              loss_name="loss")

network = Network(ops=[
    ModelOp(inputs="x", model=model, outputs="y_pred"),
    SparseCategoricalCrossentropy(y_true="y", y_pred="y_pred", outputs="loss")
])

traces = [
    UMap(model_name="LeNet", model_input="x", labels="y", label_dictionary=class_dictionary),
    ConfusionMatrix("y", "y_pred", num_classes),
    VisLogger(show_images="LeNet_UMap")
]

estimator = Estimator(network=network, pipeline=pipeline, traces=traces, epochs=5, log_steps=750)

```


```python
estimator.fit()
```

As the UMaps reveal, the network is learning to separate man-made objects from natural ones. It also becomes clear that classes like 'cars' and 'trucks' are more similar to one another than they are to 'airplaines'. This all lines up with human intuition, which can increase confidence that the model has learned a useful embedding. The UMAP also helps to identify classes which will likely prove problematic for the network. In this case, the 'bird' class seems to be spread all over the map and therefore is likely to be confused with other classes. This is born out by the confusion matrix, which shows that the network has only 496 correct classifications for birds (class 2) vs 600+ for most of the other classes. 

If you want the Trace to save its output into TensorBoard, simply add a Tensorboard Trace to the traces list like follows:


```python
from fastestimator.trace import TensorBoard

traces = [
    UMap(model_name="LeNet", model_input="x", labels="y", label_dictionary=class_dictionary),
    TensorBoard(write_images="LeNet_UMap")
]
```
