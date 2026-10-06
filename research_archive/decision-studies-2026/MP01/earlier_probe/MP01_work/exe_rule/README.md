# Rule-based MAPF-LNS for [Benchmarking Large Neighborhood Search for Multi-Agent Path Finding](https://github.com/ChristinaTan0704/mapf-lns-benchmark/tree/main)

This repository is directly modified based on [rule-based MAPF-LNS](https://github.com/Jiaoyang-Li/MAPF-LNS), with the addition of a RandomWalkProb heuristic. The input to this executable is a JSON file where the key is the agent ID and the value is a list of agent locations in 2D x and y coordinates (see [map-random-32-32-20-scene-1-agent-150.json](map-random-32-32-20-scene-1-agent-150.json) as an example). The delay and runtime information will be logged in the terminal. 


## Installation 
The code requires external libraries 
BOOST (https://www.boost.org/) and Eigen (https://eigen.tuxfamily.org/). 
An easy way to install the required libraries on Ubuntu:    
```shell script
sudo apt update
```
- Install the Eigen library (used for linear algebra computing)
 ```shell script
    sudo apt install libeigen3-dev
 ```
- Install the boost library
 ```shell script
    sudo apt install libboost-all-dev
 ```
    
After you have installed both libraries and downloaded the source code, 
go into the directory of the source code and compile it with CMake: 

```
cd exe/lns-rule-based
cmake .
make 
```
## Usage
To run LNS with rule-based heuristics, use the following command:

```shell
./rule-based-lns --destroyStrategy [Intersection / RandomWalk / Random / Adaptive / RandomWalkProb ] \
--uniform_neighbor 0 --neighborSize 32 \
--map random-32-32-20.map \
--state map-random-32-32-20-scene-1-agent-150.json \
--agentNum 150 \
--cutoffTime 300
```
- map (required): the .map file downloaded from the MAPF benchmark
- agentNum (required): number of agents in the current map
- state (required): path to the current state JSON file, key: agent id, value: list of agent location in 2D x, y coordinate, check [map-random-32-32-20-scene-1-agent-150.json](map-random-32-32-20-scene-1-agent-150.json) as an example
- uniform_neighbor (optional): (0) fixed nb_size specified by --neighborSize (1) nb_size sample from {2,4,8,16,32} (2) nb_size sample from 5~16
- cutoffTime (optional): run time limit for running the removal and replan of LNS


You can find more details and explanations for all parameters with:

```
./rule-based-lns --help
```

## References
[1] Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey, Sven Koenig.
Anytime Multi-Agent Path Finding via Large Neighborhood Search.
In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), pages 4127-4135, 2021.         

 

 
