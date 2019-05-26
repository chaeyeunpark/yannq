# YANNQ
C++ header-only library for neural network quantum states. 

## What is this?
This library is a simple head-only library for variational Monte-calro simulation of quantum states. 

If you are interested in more complete library that supports python binding and more complicated models, check [NetKet](https://www.netket.org/).

## Distinction to NetKet
Main features of this library are also available in NetKet. 
Stil, there are several differences in implementaion details in basic features.
First, YANNQ is a header-only libary. Thus it is much easiler to include in your own existing codes.
Static polymorphism, one of the powerful tools in C++, is also widely used to enhance to computational speed.
YANNQ also use shared memory parallel (SMP) programming model by employing OpenMP so can be faster than NetKet in a single node multi-processor set-up. 

## Warnings
This project started as a personal project and still has the only developer. So some features may not work. 
Especially, some Optimizers are not tested. 

## Contributions
If you want to contribute or ask for implementing some features, please [email](mailto:chae.yeun.park@gmail.com) to me.

## Future
I plan to rewrite the library in [rust](https://www.rust-lang.org/). Rust supports the techniques used in this library better than C++. 
