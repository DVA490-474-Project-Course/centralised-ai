User Guide
=======================

@page user_guide User Guide

Getting started
-----------------------

### Prerequisites
On Ubuntu and other Debian derivatives, CMake and Protobuf can be installed with:<br/>
```
sudo apt install build-essential cmake libprotobuf-dev protobuf-compiler
```

PyTorch can be downloaded from https://pytorch.org/ where the following options need to be selected:

- PyTorch build: Stable
- OS: Linux
- Package: LibTorch
- Language: C++/Java
- Compute Platform: CPU
- Choose download link with cxx11 ABI version

### Installation
1. Clone the repository:<br/>
```
git clone https://github.com/DVA490-474-Project-Course/centralised-ai.git
```
2. Navigate to the project directory:<br/>
```
cd centralised-ai
```
3. Open the CMakeLists.txt under repo root in a text editor and edit the path in the line:<br/>
```
set(CMAKE_PREFIX_PATH "/absolute-path-to/libtorch/share/cmake/Torch/")
```
<br/>
to contain the absolute path of .../libtorch/share/cmake/Torch in the libtorch directory that was downloaded in the prequisites section.

4. Create a build directory and navigate to it:<br/>
```
mkdir build & cd build
```
5. Build the source code:<br/>
```
cmake ..
make
```
6. Locate the binaries which should be stored in bin:<br/>
```
cd ../bin
```
7. Execute the desired binaries.
