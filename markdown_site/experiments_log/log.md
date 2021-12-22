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

