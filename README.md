# YANNQ
C++ header-only library for neural network quantum states. 

## What is this?
This library is a simple header-only library for variational Monte-Carlo simulation using an Ansatz based on restricted Boltzmann machine.

If you are interested in a complete library that supports python binding and complicated models, check [NetKet](https://www.netket.org/).

## Distinction to NetKet
Main features of this library are also available in NetKet. 
Still, there are several differences in the implementation details.
First, YANNQ is a header-only libary. Thus it might be easier to integrate your own existing codes.
Static polymorphism, one of the powerful tools in C++, is also widely used to enhance computational speed.
YANNQ employs shared memory parallel (SMP) programming model using OpenMP so can be faster than NetKet in a single node multi-processor set-up. 

## Warnings
This project started as a personal project and still has the only developer. So some features may not work. 
Especially, optimizers other than SGD, Adam, RMSProp are not tested.

## Contributions
If you want to contribute or ask for implementing some features, please [email](mailto:chae.yeun.park@gmail.com) to me. Also, feel free to issue or make a pull request.

## Future
I plan to rewrite the library in [rust](https://www.rust-lang.org/). Rust supports the techniques used in this library better than C++. 
