### Travelling Salesman Problem using dimod

Basic implementation of the TSP problem using Binary Quadratic Model from DWave dimod
BQM Function:

E(x)= 
j=0
∑
n−1
​
  
i 
1
​
 ,i 
2
​
 =0
i 
1
​
 

=i 
2
​
 
​
 
∑
n−1
​
 D 
i 
1
​
 ,i 
2
​
 
​
 ⋅x 
i 
1
​
 ,j
​
 ⋅x 
i 
2
​
 ,j+1modn
​
 +λ 
i=0
∑
n−1
​
  
​
 − 
j
∑
​
 x 
i,j
​
 +2 
j<k
∑
​
 x 
i,j
​
 x 
i,k
​
  
​
 +λ 
j=0
∑
n−1
​
 (− 
i
∑
​
 x 
i,j
​
 +2 
i<k
∑
​
 x 
i,j
​
 x 
k,j
​
 )
