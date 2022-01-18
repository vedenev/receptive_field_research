#### Decomposed initial condition formulas  
Let we have sequence of N 3x3 convolutions.  
This sequence of convolutions is same to one big ```(2*N + 1) x (2*N + 1)``` convolution.  
What small kernels values is need to have all 1.0 values in the big kernel?  
Let start with simpler case: 1d and only 2 kernels of size 3.
We have an equation:  
```[a1 a2 a3] * [b1 b2 b3] = [c1 c2 c3 c4 c5]```  
here * is convolution operation.  
c1 = c2 = c3 = c4 = c5 = 1.0  
```[a1 a2 a3] * [b1 b2 b3] = [1 1 1 1 1]```   
The task is to find a1 a2 a3 b1 b2 b3  
The convolution works in this way:  
c1 = a1 * b3  
c2 = a1 * b2 + a2 * b3  
c3 = a1 * b1 + a2 * b2 + a3 * b3  
c4 = a2 * b1 + a3 * b2  
c5 = a3 * b1  
To have a symmetry let a3 = a1 and b3 = b1  
then  
1 = a1 * b1  
Let a1 = 1  
then b1 = 1  and a3 = 1 and b3 = 1  
Now we have:  
```[1 a 1] * [1 b 1] = [1 1 1 1 1]```  
c2 = 1 = b + a  
c3 = 1 = 2 + a * b  
c4 = 1 = a + b  
Thus we have system of 2 equations:   
a + b = 1  
a * b = -1  
Let solve:  
a * (1 - a) = -1  
or  
a * (a - 1) = 1  
or  
a^2 - a - 1 = 0  
It is [golden ratio](https://en.wikipedia.org/wiki/Golden_ratio#Calculation) equation.
Solution:  
a = (1 - sqrt(5)) / 2  
b = (1 + sqrt(5)) / 2  
We have [alternative form of the golden ratio](https://en.wikipedia.org/wiki/Golden_ratio#Alternative_forms):  
```b = (1 + sqrt(5)) / 2 = 2 * cos(pi/5) = -2 * cos(2 * (2*pi/5))```  
```a = -1 / b = -2 / csc(pi/10) = -2 * sin(pi/10) ```  
or  
```a = -2 * sin((pi/2) - 2 * pi/5) = -2 * cos(2*pi/5)```  
Let generalize the formulas:  
```a = - 2 * cos(1 * phi)```   
```b = - 2 * cos(2 * phi)```  
where ```phi = 2 * pi / (2*N + 1)```
in this case N = 2  
Generalization for 1d case for any N:  
```[1  -2*cos(1 * phi) 1] * [1  -2*cos(2 * phi) 1] * ... * [[1  -2*cos(N * phi) 1] =```  
```[1 1 ... 1]```   
One can check that this formulas are correct by writing a code.  
Of cause this formula requires strong mathematical evidence. I didn't made it.  
For 2d case we need to convolve 1x3 and 3x1 kernels:  
```[[1  -2*cos(k * phi) 1]] * [[1]  [-2*cos(k * phi)] [1]] =```  
```text
[[1               -2*cos(k*phi)     1            ]
  [-2*cos(k*phi)   4*cos(k*phi)**2   -2*cos(k*phi)]
  [1               -2*cos(k*phi)     1            ]]
``` 
  





 
  