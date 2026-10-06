# Prioritize Planning (PP) Replan for [Benchmarking Large Neighborhood Search for Multi-Agent Path Finding](https://github.com/ChristinaTan0704/mapf-lns-benchmark/tree/main)


The PP replan function is extracted from [rule-based MAPF-LNS](https://github.com/Jiaoyang-Li/MAPF-LNS). The input to the executable is a JSON file containing the complete feasible solution and a list of agents to replan. The algorithm maintains the paths of the remaining agents unchanged, treating them as space-time obstacles, and only replans the paths for the agents in the removal set using the prioritize planning (pp) algorithm.



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
    
After you installed both libraries and downloaded the source code, 
go into the directory of the source code and compile it with CMake: 

```
cmake .
make 
```
## Usage

**Step 1**: Start the PP replan program.

```shell
./pp_open \
--map random-32-32-20.map \
--state map-random-32-32-20-scene-1-agent-150.json \
--agentNum 150 
```

- map (required): the .map file downloaded from the MAPF benchmark
- state (required): path to the current state JSON file, key: agent id, value: list of agent location in 2D x, y coordinate, check [map-random-32-32-20-scene-1-agent-150.json](map-random-32-32-20-scene-1-agent-150.json) as an example
- agentNum (required): number of agents in the current map
- replanAgents (required): list of agents to replan
- cutoffTime (optional): run time limit for running the removal and replan of LNS

You can find more details and explanations for all parameters with:
```
./pp_open --help
```

**Step 2** : Input the current state JSON file and the list of replan agents to get the removal and replan information 

```
--state map-random-32-32-20-scene-1-agent-150.json --replanAgents 3 25 62 117 134 
```

If the cost of replanned paths improves (improvement > 0), the program will output the newly planned paths for the replanned agents.

## References
[1] Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey, Sven Koenig.
Anytime Multi-Agent Path Finding via Large Neighborhood Search.
In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), pages 4127-4135, 2021.         

 

 
