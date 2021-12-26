# Experiments log
accuracy - number of correct unswers devided by number of answers  
depth - depth of a neural net, number of convolutional layers  
distance - distance in pixels from center of the input image where e symbol is paced and diaeresis  
white color - no data  
  
#### 1. No residual connection net  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16     
[experiment_field_size_vs_depth.py](../../experiments/experiment_field_size_vs_depth.py)    
![no res connections](./field_size_vs_depth_no_res_connections.png)  
receptive field size is O(sqrt(N)), where N is number of layers  
after depth = 40 train are not stable probably because of [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)  
    
#### 2. Residual connections net
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16     
[experiment_field_size_vs_depth_res.py](../../experiments/experiment_field_size_vs_depth_res.py)    
![with res connections](./field_size_vs_depth_with_res_connections.png)  
receptive field size is O(sqrt(N))  
    
#### 3. Special initial condition
each kernel has xavier initial condition only for the upper row of the kernel. All rest elements of the kernel is close to 0  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16     
[experiment_field_size_vs_depth_res.py](../../experiments/experiment_field_size_vs_depth_res.py)    
is_shifted_init=True  
![special init condition](./field_size_vs_depth_special_init_condition.png)  
receptive field size is O(N)  
    
#### 4. Residual connections net, deeper
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16  
is_shifted_init=False  
[experiment_field_size_vs_depth_res.py](../../experiments/experiment_field_size_vs_depth_res.py)    
![with res connections, deeper](./field_size_vs_depth_with_res_connections_additional.png)  
receptive field size is O(sqrt(N))  
    
#### 5. Get field size by forward pass
[experiment_field_size_by_forward_pass.py](../../experiments/experiment_field_size_by_forward_pass.py)  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16  
net: NoPoolsNetRes  
is_shifted_init=False  
is_show_field=True  
![field size by forward pass](./field_size_vs_depth_by_forward_pass.png)  
receptive field size is O(sqrt(N))  
  
#### 6. Get field size by forward pass, constant weights init
[experiment_field_size_by_forward_pass_constant.py](../../experiments/experiment_field_size_by_forward_pass_constant.py)  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16  
net: NoPoolsNet  
is_constant_init=True   
![field size by forward pass constant](./field_size_vs_depth_by_forward_pass_constant.png)  
receptive field size is O(sqrt(N))  
output value along curve y= 3 * sqrt(N):  
![field size by forward pass constant level](./field_size_vs_depth_by_forward_pass_constant_level.png)  
the value is about 0.0012  
  
#### 7. Decomposed initial condition, field size by forward pass
[experiment_field_size_vs_depth_res_decomposed_init.py](../../experiments/experiment_field_size_vs_depth_res_decomposed_init.py)  
kernel initialized with special values that is decomposition of convolution witch kernel is m x m matrix of ones.     
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16  
[initializers.py/decomposed_init](../../initializers.py#L4)    
![decomposed init by pass](./field_size_vs_depth_by_forward_pass_decomposed_init.png)    
receptive field size is O(N) until depth = 14  
  
#### 8. Decomposed initial condition
[experiment_field_size_vs_depth_res_decomposed_init.py](../../experiments/experiment_field_size_vs_depth_res_decomposed_init.py)  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16  
![decomposed init](./field_size_vs_depth_decomposed_init.png)  
no recognition at all, all accuracies is about 0.5  
  
#### 9. Circular initial condition
[experiment_field_size_vs_depth_dot_circular.py](../../experiments/experiment_field_size_vs_depth_dot_circular.py)  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16  
[initializers.py/circular_init](../../initializers.py#L51)  
circular amplitude: 0.2  
angle: 360 degrees  
![circular init](./field_size_vs_depth_circular_init_0_2_decemated_less.png)   
  
#### 10. Circular initial condition, no circular init condition, just xavier init
[experiment_field_size_vs_depth_dot_circular.py](../../experiments/experiment_field_size_vs_depth_dot_circular.py)  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16  
angle: 360 degrees  
![circular init, no](./field_size_vs_depth_circular_init_no_circ_init.png)  
  
#### 11. Circular initial condition, version 2, 0.6 no circular init condition, just xavier init
[experiment_field_size_vs_depth_dot_circular.py](../../experiments/experiment_field_size_vs_depth_dot_circular.py)  
convolutional layer kernel size: 3x3  
number of intermidiate featuremaps: 16
DECREASE_FACTOR = 0.6  
angle: 360 degrees  
[initializers.py/circular_init_version_2](../../initializers.py#L131)  
![circular init v2](./field_size_vs_depth_circular_init_v2_0_6.png)  
  
  




