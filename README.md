# CuBOIDS

Explorative sandbox for GPU accelerated C and CUDA, with Boids "bird-oid objects" simulation.

## Getting Started

## Roadmap

- Structures for BOID representation:
   - x, y, z, dx, dy, dz
- CUDA Kernel Functions:
   - Get random initialisation of Positions from the CPU in the beginning.
   - GetNeighbours
   - ParallelSort
   - BOID
      - sensory information: near wall or boundary, close BOIDS with their distance/orientation/velocity
      - friendly or predatory behavior? Avoidance or Follow?
- Data storage for Output:
   - Plain text for now, or binary, nothing fancy.
- Data display:
   - External Python
 
## The program flowchart

```mermaid
flowchart
    subgraph GPU with CUDA
        direction TB
        GPU_1[Variables]
        subgraph BOID Logic
        GPU_6[basic logic function]
        GPU_2[find_neighbour]
        GPU_3[hashtable]
        GPU_4[sort]
        GPU_5[boundary check]
        GPU_7[FriendOrFoe check]
        end
        GPU_6 --> GPU_2
        GPU_6 --> GPU_5
        GPU_6 --> GPU_7
        GPU_2 --- GPU_3
        GPU_3 --- GPU_4
        GPU_6 --read--> GPU_1
        GPU_6 --update-->GPU_1
    end
    subgraph CPU
        direction TB
        CPU_1[Initalisation]
        CPU_2[Variables]
        CPU_3[save in file]
        CPU_1 --> CPU_2
        CPU_2 --after update-->CPU_3
    end
CPU_2 --initialise--> GPU_1
CPU_2 --get update--> GPU_1
```

## Class and function overview

<!-- for documentation of mermaid editor https://mermaid.js.org/syntax/classDiagram.html -->
```mermaid
classDiagram
    direction LR
    class BOID{
        +float position x, y, z
        +float velocity dx, dy, dz
        +int GridID gx, gy, gz
        +int species
        +boid_logic()
        +check_obstacle()
    }

    note for HashtableLookup "implementation depends if\n functionality is external\n or if BOIDS itself is saved in a grid cell"
    class HashtableLookup{
        
    }
```

## Console Output

for 1000000 boids:

'''
number of boids: 1000000 , number of bins: 100
thread: 896
blocks: 1117
Boid 0 - Position: (9.700000, 5.700000, 0.000000)__HASH: 94
Boid 1 - Position: (5.000000, 0.300000, 0.000000)__HASH: 93
Boid 2 - Position: (4.600000, 2.700000, 0.000000)__HASH: 0
Boid 3 - Position: (1.100000, 0.200000, 0.000000)__HASH: 0
Boid 4 - Position: (9.200000, 8.500000, 0.000000)__HASH: 94
Boid 5 - Position: (3.000000, 7.000000, 0.000000)__HASH: 63
Boid 6 - Position: (4.000000, 7.600000, 0.000000)__HASH: 63
Boid 7 - Position: (9.900000, 6.200000, 0.000000)__HASH: 94
Boid 8 - Position: (4.500000, 8.900000, 0.000000)__HASH: 63
Boid 9 - Position: (6.700000, 5.300000, 0.000000)__HASH: 94
0.389959 ms | 389.959 µs | 2564.37 it/s
bins: 
40563272  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 189  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 63  252  _ _ _ _ _ 
'''

for 100000 boids:

'''
number of boids: 100000 , number of bins: 100
thread: 896
blocks: 112
Boid 0 - Position: (9.700000, 5.700000, 0.000000)__HASH: 94
Boid 1 - Position: (5.000000, 0.300000, 0.000000)__HASH: 93
Boid 2 - Position: (4.600000, 2.700000, 0.000000)__HASH: 0
Boid 3 - Position: (1.100000, 0.200000, 0.000000)__HASH: 0
Boid 4 - Position: (9.200000, 8.500000, 0.000000)__HASH: 94
Boid 5 - Position: (3.000000, 7.000000, 0.000000)__HASH: 63
Boid 6 - Position: (4.000000, 7.600000, 0.000000)__HASH: 63
Boid 7 - Position: (9.900000, 6.200000, 0.000000)__HASH: 94
Boid 8 - Position: (4.500000, 8.900000, 0.000000)__HASH: 63
Boid 9 - Position: (6.700000, 5.300000, 0.000000)__HASH: 94
0.128836 ms | 128.836 µs | 7761.81 it/s
bins: 
477208  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 3  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 1  4  _ _ _ _ _ 
'''


## References

__Bibliography__

- <https://doi.org/10.2312/cgvc.20191258>
- [Rama C, Hoetzlein, Graphics Devtech, NVIDIA, FAST FIXED-RADIUS NEAREST NEIGHBORS: INTERACTIVE MILLION-PARTICLE FLUIDS](https://on-demand.gputechconf.com/gtc/2014/presentations/S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf)
- <https://doi.org/10.1111/j.1467-8659.2010.01832.x>

__CUDA - Lecture notes and introduction__

- [Pennsylvania Lecture plan and slides](https://cis565-fall-2021.github.io/syllabus/)
- [Instrucution set for the Lecture course on CUDA flocking](https://github.com/CIS565-Fall-2023/Project1-CUDA-Flocking/blob/main/INSTRUCTION.md)
- [Pennsylvania 2022 Lecture introduction](https://github.com/CIS565-Fall-2022/Project1-CUDA-Flocking/blob/main/INSTRUCTION.md)
- [Pennsylvania Assignment DONE](https://github.com/AmanSachan1/CUDA-Boid-Flocking/tree/master)
- [caltech lecture with notes](http://courses.cms.caltech.edu/cs179/)
- [Pennsylvania hardware setup](https://cis565-fall-2022.github.io/setup/)
- [Hardware setup for Linux](https://cis565-fall-2022.github.io/setup-linux/)
- [NVIDIA: cuCollections header-only library of GPU-accelerated, concurrent data structures](https://github.com/NVIDIA/cuCollections#data-structures)
- [NVIDIA: Maximizing Performance with Massively Parallel Hash Maps on GPUs](https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/)

__BOIDS - Introduction and Logic__

- [Paper on BOIDS revisited](https://www.tandfonline.com/doi/full/10.1080/13873950600883485)
- [online introduction to BOIDS](https://betterprogramming.pub/mastering-flock-simulation-with-boids-c-opengl-and-imgui-5a3ddd9cb958)

## Licence

Distributed under the GNU GENERAL PUBLIC LICENSE. See LICENSE for more information.
