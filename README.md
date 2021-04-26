# YANNQ
C++ header-only library for neural network quantum states. 

## What is this?
This library is a simple header-only library for variational Monte-Carlo simulation using an Ansatz based on restricted Boltzmann machine.

If you are interested in a complete library that supports python binding and complicated models, check [NetKet](https://www.netket.org/).

## Distinction to NetKet (version 2)
Main features of this library are also available in NetKet. 
Still, there are several differences in the implementation details.
First, YANNQ is a header-only libary. Thus it might be easier to integrate your own existing codes.
Static polymorphism, one of the powerful tools in C++, is also widely used to enhance computational speed.
YANNQ employs shared memory parallel (SMP) programming model using Intel TBB so can be faster than NetKet in a single node multi-processor set-up. 

> NetKet has fully moved to Python (using JAX) in their version 3.0. 

## Warnings
This project started as a personal project and still has the only developer. So some features may not work. 
Especially, optimizers other than SGD, Adam, RMSProp are not tested.

## Compiler supports
Currently, yannq compiles in gcc (version >= 5), clang (only version 3.8 is tested), and intel C compiler under C++17 support. 

> In general, we have observed that intel C compiler generates more efficient binary codes (around ~30% speedup). 

## Contributions
If you want to contribute or ask for implementing some features, please [email](mailto:chae.yeun.park@gmail.com) to me. Also, feel free to issue or make a pull request.

## Future
I will plan to use this library only for legacy codes. As Google JAX naturally supports GPUs and offers flexible network designs, it is much better to use JAX based libraries when one has GPU resources.
