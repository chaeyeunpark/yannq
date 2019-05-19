# YANNQ

C++ head only library for neural network quantum states. 

## Name and history
YANNQ stands for "Yet another neural network quantum (library)". 
This is named as there is a project for this purpose named [NetKet](https://www.netket.org/).
I started to code YANNQ from August 2018 when I din't aware NetKet. 
I found NetKet from github around Oct. 2018 but it didn't have enough features at that time so I continued to develop my one implementation.

## Distinction to NetKet
Main features of this library are also available in NetKet. 
Stil, there are several differences in implementaion details.
First, YANNQ is a head-only libary. Thus it is much easiler to add your own code into it. 
Static polymorphism, one of the powerful tools in C++, is also widely used to enhance to computational speed.
YANNQ also use shared memory parallel (SMP) programming model by employing OpenMP so may be faster than NetKet in a single node multi-processor set-up. 


## Warnings
This project started as a personal project and still has the only developer. So some features may not work. 
Especially, some Optimizers are not tested. 


## Contributions
Pull requests are welcome.

## Future
I plan to rewrite the library in [rust](https://www.rust-lang.org/). Rust supports the techniques used in this library better than C++. 
