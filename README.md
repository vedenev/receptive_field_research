# Receptive field research

[experiments log](./markdown_site/expereiments_log.md)  
  
The research is about understanding of dependence between fully convolutional neural net depth and receptive field size. A lot of experiments were done to find the dependence. See full [experiments log](markdown_site/expereiments_log.md) for details.  
  
#### 1. What is the receptive field?
The concept came from biology:  
[wikipedia: receptive field: In the context of neural networks](https://en.wikipedia.org/wiki/Receptive_field#In_the_context_of_neural_networks).  
The fully convolutional neural net input is image. Output is image or images too.  
Let select a pixel on the output image (blue, see image below). Only limited set of pixel (red) on the input image can influence the selected pixel.  
  
![filed](./markdown_site/picture_receptive_field_size.png)  
  
This region of red pixels is the receptive field.   
Size of this filed is important in detection tasks.  
The object to be detected should be within this field.  
Otherwise some important features of the object will be outside the field and will be not used.  
So quality of the detection can get worse.  

#### 2. Theoretical receptive field size, naive  
Let see a convolutional layer. It consist of convolutions.  
Let see a kernel of size s. It can shift a pixel at half size of the kernel at the maximum:  
  
![conv shift](./markdown_site/picture_convolution_shift.png)   
  
maximal shift is (s - 1) / 2
for 3x3 kernel: (3 - 1) / 2 = 1   
If we have sequence of N 3x3 convolution layers then maximal shift will be:  
N * (s - 1) / 2 = N  
So field size increase as O(N) with number of layers.  
As we can have pooling layers in the net then featuremaps resolution decrease.  
It can be considered as increasing pixels sizes.  
So if we have convolution layer after first 2x2 polling layer then size of the convolution layer should be increase in 2 times.  
And the maximal shift will be (2 * s - 1) / 2

#### 3. Theoretical receptive field size
It can be shown that field size increase as O(sqrt(N)) with number of layers.  
Let consider 1d case.
Convolution is used in probability distribution:  
[wikipedia: Convolution of probability distributions](https://en.wikipedia.org/wiki/Convolution_of_probability_distributions)    
So let consider the kernel as a distribution.  
Let the convolution kernel have all elements are positive numbers.    
Let we have N successive convolutions of size s.
So the final distribution is convolution of convolutions of convolutions etc.  
According to the wikipedia article the final distribution is distribution of sum of random variables.  
Form the other had we have [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem)  
It says that the distribution of sum is normal distribution with parameters:  
expected value: mu = sum(mu_i)  
variance: sigma^2 = sum(sigma_i^2)  
Let considered that elements of the kernels has no special shifts so mu_i = 0
Size of the kernel is 3x3 so sigma_i = 3  
Then size of the final distribution is sqrt(sigma^2) = sigma = sqrt(sum(3^2)) = O(sqrt(N))  
Please note: there is no any random variables inside the convolution.  
Theory of probability was used just to get the final O(sqrt(N)).  
In case of the net with pool we need to increase kernel sizes.  
For example we have a net: 3x3 convolution - 2x2 pool - 3x3 convolution, then we need to use:  
```sqrt(3^2 + (3*2)^2)```  
here we have 3*2 because after pooling featuremaps has decreased in 2 times resolution.  

  
#### 4. e dataset and no pooling net  
A set of experiments was think out to check the theoretical field size.
Fully convolutional neural net was used. For simplicity, it has no pooling layers.  
The code of the net: [no_pools_net_res.py](./nets/no_pools_net_res.py)  
Special synthetic dataset was used:  
It can be considered as OCR task (optical character recognition) with only 2 possible letters:  
е  
ё  
Russian letter е can have diaeresis. The diaeresis is double dot above the letter.  
Russian е with the diaeresis is another letter.  
e always has fixed position in the center of the image.  
Distance between e and the diaeresis is adjustable.  
It is detection task. The task is to find center of e.  
There are 2 featuremaps on the output of the nets.  
One for е and one for ё.  
Center of the letter encoded with small gaussian spot.  
![e dataset](./markdown_site/picture_e_dataset.png)   
First row is input image, second row is featuremap for е, third row is featuremap for ё.  
Bigger distance:
![e dataset big dist](./markdown_site/picture_e_dataset_bigger_distance.png)   
The task of the net is to predict to right spot.  
How the е/ё decission is made:  
if mean value of small area in the center of the output е-featuremap is higher
than in ё-featuremap then it is e symbol, ё otherwise.  
Accuracy is calculated.   
Accuracy is number of correct answers divided by number of all answers.   
If the receptive field size of the net will be too small, than the accuracy expected to be close to 0.5 (random guess).  
If the size is high enough then the accuracy is close to 1.0.  
That is how the receptive field size can be found:  
we can scan distance to diaeresis until the accuracy drops from 1.0 to 0.5.  
The dataset code: [e_symbol_dataset.py](./dataset_generator/e_symbol_dataset.py)  
There are more difficult dataset also:  
![e dataset dot](./markdown_site/picture_e_dataset_dot.png)  
Here the diaeresis is a dot.  
And the dot can be at any place around the e at fixed predefined distance.
The dataset code: [e_symbol_dot_dataset.py](./dataset_generator/e_symbol_dot_dataset.py)  
This dataset has higher variability. 
Small scale elastic augmentation are used, see: [augmentator.py](./dataset_generator/augmentator.py)  

 
  


     
