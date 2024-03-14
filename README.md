# CuBOIDS

Explorative sandbox for GPU accelerated C and CUDA, with Boids "bird-oid objects" simulation.

## Getting Started

## Roadmap
- Structures for BOID representation:
   - x, y, dx, dy
- CUDA Kernel Functions:
   - Get random initialisation of Positions from the CPU in the beginning. Do not want to programm a pseudo random generator in CUDA... although can probablby easiliy be done with atomic instructions... if supported.
   - GetNeighbours
   - ParallelSort
   - BOID
      - sensory information: near wall or boundary, close BOIDS with there distance/orientation/velocity
         - firendly or predetory behaviour? Avoidance or Follow?
- Datastorage for Output:
   - Plain text for now, or binary, nothing fancy.
- Datadisplay:
   - External Python
 
## Example Flowchart
Can i label this???

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
CPU_2 --initalise--> GPU_1
CPU_2 --get update--> GPU_1
```


## Licence

Distributed under the GNU GENERAL PUBLIC LICENSE. See LICENSE for more information.
