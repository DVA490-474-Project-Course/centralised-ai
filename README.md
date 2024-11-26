Centralised AI
=======================

About
-----------------------
This repository contains the executable code intended for running on the central computer, which is responsible for the overarching control of the robot team, as well as functionality for receiving data from the shared vision system ([SSL-Vision](https://github.com/RoboCup-SSL/ssl-vision)) and referee software ([SSL Game Controller](https://github.com/RoboCup-SSL/ssl-game-controller)) of RoboCup SSL, as well as fuctionality for communicating with the simulation software grSim (https://github.com/RoboCup-SSL/grSim).

### Built with
The Centralized AI software is built with the following:

- [CMake](https://cmake.org/)
- [Protobuf](https://protobuf.dev/)
- [PyTorch](https://pytorch.org/)

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

Usage
-----------------------

Roadmap
-----------------------
API:
- [x] Develop Simulation interface
- [x] Develop SSL interface

Collective Robot Behaviour:
- [] Develop AI algorithm
- [] Ability to send and receive data from simulation
- [x] Ability to receive referee commands from SSL interface
- [x] Ability to receive pose data from SSL interface
- [] Ability to send commands to robot
- [] Ability to receive information from robot


Design diagrams
-----------------------
Design diagrams/files can be found in the [docs](/docs) directory. Additionally they are available on:
- [Simulation interface](https://www.mermaidchart.com/raw/16fc3609-d826-440a-bef5-40a7a39f1140?theme=dark&version=v0.1&format=svg)
- [SSL interface](https://www.mermaidchart.com/raw/6428d81b-020c-4506-a2c5-c319e514648f?theme=dark&version=v0.1&format=svg)
- [Collective Robot Behaviour](https://www.mermaidchart.com/raw/80201d9e-191f-4971-82e5-fbe7ab0a4692?theme=light&version=v0.1&format=svg)


License
-----------------------
Distributed under the MIT License. See [License](/LICENSE) for more information.

Contributors and contact
-----------------------
- Jacob Johansson:jjn20030@student.mdu.se
- Viktor Eriksson:ven20002@student.mdu.se
- Aaiza Aziz Khan: akn23018@student.mdu.se
- Carl Larsson: cln20001@student.mdu.se
- Shruthi Puthiya Kunnon: spn23001@student.mdu.se
- Emil Åberg: eag24002@student.mdu.se
