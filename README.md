# ImageNet-Analysis-ResNet18-AlexNet-and-Vgg16-on-A100-and-T4

### Experiment design

In this experiment, I selected three neural network models: ResNet18, AlexNet, and VGG16, and tested them in cloud environments with different GPUs: NVIDIA A100-SXM4-40GB and NVIDIA Tesla T4. The focus of our study is the time and space complexity of processing a single batch (batch size = 256) during a single forward pass.

I first used the external library `torchinfo` to estimate the FLOPs and memory usage of the three models, then approximated the runtime based on the estimated result and GPU specifications. Then I measured the runtime and memory usage under different GPUs and profiled the FLOPs using *Nsignt Compute*. Finally, I plotted the Roofline Model based on the GPU specifications and conducted a comparative analysis of the experimental results.



### Complexity estimation

To estimate the complexity, I utilized the `summary` function from the Python `torchinfo` library (https://github.com/TylerYep/torchinfo). This function provides a visualization of the model and estimates MACs (multiply-add operations), input data size, parameter size, and intermediate result sizes. The usage is as follows:

```python
from torchinfo import summary
print(summary(model, input_size=(args.batch_size, 3, 224, 224)))
```

**Note:** `torchinfo` needs to be installed and run in a non-Singularity environment.



The visualization result for **ResNet18** is shown below:

<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/resnet18.png" alt="resnet18" style="zoom: 45%;" />

We can use the total multi-adds value to estimate FLOPs. A single multi-add consists of one addition and one multiplication, equivalent to two floating-point operations:

$Estimated\ FLOPs = 2 * \#Total\ multi-adds = 2 * 464.41G = 928820000000$



The memory usage can be estimated based on the input size, forward/backward pass size, and parameter size. Note that the forward/backward pass size includes both the activations during the forward pass and gradients during the backward pass, so it should be divided by two to estimate the memory occupied by activations during the forward pass. The total estimated memory usage is:

$Estimated\ memory = \#input + \#params + \#forward\_backward\_pass\_size/2 \\= 154.14 MB + 46.76MB + 10175.33MB/2 = 5288565000\ Bytes$



We estimate the runtime on both GPUs using the following formula:

<img src="/Users/bang/Desktop/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7/Screenshot%202024-11-18%20at%201.59.32%20PM.png" alt="Screenshot 2024-11-18 at 1.59.32 PM" style="zoom:40%;" />

For the NVIDIA A100-SXM4-40GB, based on its GPU specs, the peak FLOPS (FP16 precision) is 312 TFLOPS, and the peak memory bandwidth is 1555 GB/s. The estimated runtime is:

$Estimated\ time = max(5288565000/ (1555*10^9), 928820000000/(312*10^{12})) \\= max(0.00340101, 0.00297699) = 0.00340101\ s$





For the NVIDIA Tesla T4, based on its GPU specs, the peak FLOPS (single precision) is 8.1 TFLOPS, and the peak memory bandwidth is 300 GB/s. The estimated runtime is:

$Estimated\ time = max(5288565000/ (300*10^9), 928820000000/(8.1*10^{12})) \\= max(0.01762855, 0.11466914) = 0.11466914\ s$



The visualization result for **AlexNet**:

<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/alexnet.png" alt="alexnet" style="zoom:45%;" />

The estimation process for FLOPs, memory, and runtime is the same as above:

$Estimated\ FLOPs = 2 * \#Total\ multi-adds = 2 * 182.96G = 365920000000$

$Estimated\ memory =  \#input + \#params + \#forward\_backward\_pass\_size/2 \\= 154.14 MB + 244.40MB + 1012.09MB/2 = 904585000\ Bytes$



For the NVIDIA A100-SXM4-40GB, the estimated runtime is:

$Estimated\ time = max(904585000/ (1555*10^9), 365920000000/(312*10^{12})) \\= max(0.00058173, 0.00117282) = 0.00117282\ s$



For the NVIDIA Tesla T4, the estimated runtime is:

$Estimated\ time = max(904585000/ (300*10^9), 365920000000/(8.1*10^{12})) \\= max(0.00301528, 0.04517531) = 0.04517531\ s$



The visualization result for **VGG16**:

<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/vgg16.png" alt="vgg16" style="zoom:50%;" />

The estimation process for FLOPs, memory, and runtime is the same as above:

$Estimated\ FLOPs = 2 * \#Total\ multi-adds = 2 * 3.96T = 7920000000000$

$Estimated\ memory =  \#input + \#params + \#forward\_backward\_pass\_size/2 \\= 154.14 MB + 555.43MB + 27764.15MB/2 = 14591645000\ Bytes$



对于NVIDIA A100-SXM4-40GB, 

$Estimated\ time = max(14591645000/ (1555*10^9), 7920000000000/(312*10^{12})) \\= max(0.00938369, 0.02538462) = 0.02538462\ s$



对于NVIDIA TESLA T4，

$Estimated\ time = max(14591645000/ (300*10^9), 7920000000000/(8.1*10^{12})) \\= max(0.04863882, 0.97777778) = 0.97777778\ s$



**The table below summarizes all the estimated results for the three models:**

|              | Estimated FLOPs | Estimated memory usage (Bytes) | Estimated time on A100 (seconds) | Estimated time on T4 (seconds) |
| :----------: | :-------------: | :----------------------------: | :------------------------------: | :----------------------------: |
| **resnet18** |  928820000000   |           5288565000           |            0.00340101            |           0.11466914           |
| **alexnet**  |  365920000000   |           904585000            |            0.00117282            |           0.04517531           |
|  **vgg16**   |  7920000000000  |          14591645000           |            0.02538462            |           0.97777778           |



### Complexity measurement

We used the `time` module to measure the time taken for a single batch during a forward pass and `torch.cuda.memory_allocated` to measure the memory usage before and after the forward pass. The code is as follows:

```python
def train(train_loader, model, criterion, optimizer, epoch, device, args):
    ...
    for i, (images, target) in enumerate(train_loader):
    	...
        # compute output
        mem_before = torch.cuda.memory_allocated()
        torch.cuda.synchronize()  
        start_time = time.monotonic()
        output = model(images)
        torch.cuda.synchronize()
        end_time = time.monotonic()
        mem_after = torch.cuda.memory_allocated()
        print(f"Time: {end_time - start_time:.6f} seconds, "
              f"Memory: {mem_after - mem_before} bytes")
        quit()
        ...
```

Here is an emample result of ResNet18 on the NVIDIA A100:

<img src="/Users/bang/Desktop/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7/Screenshot%202024-11-18%20at%202.57.17%20PM.png" alt="Screenshot 2024-11-18 at 2.57.17 PM" style="zoom:60%;" />



I used *Nsight Compute* to profile the counts of four types of floating-point operations (fadd, fmul, ffma, and fp16) and exported the results to a CSV file. The command used is as follows (using ResNet18 as an example):

```bash
ncu --profile-from-start off \
    --metrics smsp__sass_thread_inst_executed_op_fadd_p.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fp16_pred_on.sum \
    --target-processes all \
    --csv > report.csv \
    python3 main.py -a resnet18 --dummy
```

The corresponding modifications in `main.py` are as follows:

```python
import torch.cuda.profiler as ncu

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    ...
    for i, (images, target) in enumerate(train_loader):
    	...
        # compute output
        ncu.start()  
        output = model(images)
        ncu.stop()    
        quit()
        ...
```

Due to the outdated `ncu` version in the Singularity environment, which does not support aggregate mode, a custom Python script (`aggregate.py`) is used to read the CSV file and aggregate the results of all kernels (noting that the `ffma` count should be multiplied by 2). The final aggregated results are as follows (using ResNet18 on the NVIDIA A100 as an example):

<img src="/Users/bang/Desktop/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7/Screenshot%202024-11-18%20at%203.06.16%20PM.png" alt="Screenshot 2024-11-18 at 3.06.16 PM" style="zoom:67%;" />



**Following the above process, I obtained the performance results of the three models on two different GPUs.** **The summarized results are shown in the table below:**

NVIDIA A100-SXM4-40GB :

|              |    FLOPs     | Memory Usage (Bytes) | Time (seconds) |
| :----------: | :----------: | :------------------: | :------------: |
| **resnet18** | 90564086824  |      5529157632      |    3.074327    |
| **alexnet**  | 81600192512  |      796762112       |    3.020387    |
|  **vgg16**   | 258692444160 |     18619539456      |    3.430079    |

NVIDIA Tesla T4:

|              |    FLOPs     | Memory Usage (Bytes) | Time (seconds) |
| :----------: | :----------: | :------------------: | :------------: |
| **resnet18** | 397410572608 |      5524963328      |    2.845695    |
| **alexnet**  | 150608494592 |      796762112       |    3.020387    |
|  **vgg16**   |     N/A      |  cuda out of memory  |      N/A       |

Note that VGG16 requires excessive memory for intermediate activations during the forward pass, causing a CUDA out-of-memory error on the Tesla T4 (with 16GB GDDR6). As a result, FLOPs, memory usage, and runtime could not be measured.



### Roofline modeling

The specifications for the NVIDIA A100-SXM4-40GB and NVIDIA Tesla T4 are as follows:

<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/A100.png" alt="A100" style="zoom:50%;" />



<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/T4.png" alt="T4" style="zoom:50%;" />

The peak FLOPS and peak memory bandwidth data can be organized into the following table:

|                       | FP16 FLOPS | Memory bandwidth |
| :-------------------: | :--------: | :--------------: |
| NVIDIA A100-SXM4-40GB | 312 TFLOPS |    1555 GB/s     |
|       NVIDIA T4       | 8.1 TFLOPS |     300 GB/s     |



For the NVIDIA A100-SXM4-40GB, its roofline before reaching the ridge point can be approximated as $y=kx$, where $x$ is the Arithmetic Intensity (FLOP/byte), $y$ is the performance (FLOP/s), and $k=1555 GB/s$. To compute the ridge point:

$312TFLOPS = 1555GB/s* x_{ridge} \Rightarrow x_{ridge}\approx186.96\ FLOP/byte$



Taking ResNet18 on the A100 as an example, based on the measured results from the previous section, we find:

$AI = \#FLOPs/\#memory\_usage=90564086824/5529157632\approx 16.38\ FLOP/byte$

$FLOPS=\#FLOPs/time=90564086824/3.074327\approx0.029\ TFLOPS$



Folloing the process, we can calculate **the achieved values for all three models, summarized in the table below:**

|              | x: Arithmetic Intensity (FLOP/byte) on A100 | y: Performance (FLOPS) on A100 |
| :----------: | :-----------------------------------------: | :----------------------------: |
| **resnet18** |                    16.38                    |          0.029 TFLOPS          |
| **alexnet**  |                   102.41                    |          0.027 TFLOPS          |
|  **vgg16**   |                    13.89                    |          0.075 TFLOPS          |

 

The corresponding roofline plot (logarithmic scale) is shown below:

<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/roofline_A100.png" alt="roofline_A100" style="zoom: 40%;" />



Similarly, for the NVIDIA T4, the roofline before the ridge point is $y=300GB/s*x$,   $x_{ridge}=27\ FLOP/byte$ 

**The achieved values for the three models on the T4 roofline model are summarized in the table below:**

|              | x: Arithmetic Intensity (FLOP/byte) on T4 | y: Performance (FLOPS) on T4 |
| :----------: | :---------------------------------------: | :--------------------------: |
| **resnet18** |                   71.93                   |         0.140 TFLOPS         |
| **alexnet**  |                  189.03                   |         0.050 TFLOPS         |
|  **vgg16**   |                    N/A                    |             N/A              |



对应的roofline图如下：

<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/roofline_T4.png" alt="roofline_T4" style="zoom:40%;" />





### Discussion

- Comparing the estimated FLOPs with the measured FLOPs, we find that the estimated values are significantly higher. This is likely because NVIDIA GPUs have different types of Tensor Cores that switch between precisions to maximize computational efficiency. The four metrics captured by `ncu` are insufficient to account for all floating-point operations. The difference in measured FLOPs between the A100 and T4 is likely due to similar reasons, as well as differences in Tensor Core support. For example, on the A100, a portion of floating-point operations are performed in FP16, while on the T4, this portion is zero (as shown in the figure below).

<img src="/Users/bang/Desktop/Cloud%20and%20ML/proj1/resnet18_T4_flops.png" alt="resnet18_T4_flops" style="zoom:50%;" />

- Comparing the estimated memory usage with the measured memory usage, we find that they are generally consistent. The differences can be attributed to the simplification in our memory usage estimation, where we assumed the forward/backward pass size is evenly divided for forward activations, whereas in reality, the ratio is not strictly 1:1. The measured memory usage is nearly identical on the A100 and T4, indicating that the measurements are accurate. For VGG16, the estimated memory usage is 13.6GB, while the measured value on the V100 is approximately 17.3GB. This explains the CUDA out-of-memory error on the T4, which only has 16GB of memory.
- Comparing the estimated runtime with the measured runtime, we observe that the measured time is several orders of magnitude higher. This is likely because a single forward pass of a batch size of 256 is insufficient to fully utilize the GPU’s parallelism. Instead, kernel launch overhead, computation latency, and memory latency dominate. The consistent runtime of around 3 seconds across measurements reflects this. On the A100, I measured the runtime for ResNet18 with batch sizes of 1 and 1024, resulting in runtimes of 2.337300 seconds and 5.331261 seconds, respectively. The lack of a linear relationship with batch size confirms that the small workload leads to significant measurement inaccuracies.
- Comparing the roofline models of the two GPUs, it is reasonable that the ridge point of the A100-SXM4-40GB shifts further right compared to the T4. This is because the A100 offers nearly 40 times higher FLOPS but only 5 times higher memory bandwidth, resulting in a higher arithmetic intensity for perfectly balanced computation and data movement.
- The arithmetic intensity(AI) of the same model differs across GPUs because AI is calculated based on the ratio of measured FLOPs to memory, and the measured FLOPs are inaccurate and insufficient to represent the true FLOPs. Performance is also calculated using the measured FLOPs divided by the measured runtime, both of which are inaccurate. The measured runtime, in particular, is likely much greater than the actual execution time, leading to extremely low performance values for all three models. This issue could be mitigated by increasing the batch size or measuring over multiple batches or even epochs to better utilize GPU parallelism.
