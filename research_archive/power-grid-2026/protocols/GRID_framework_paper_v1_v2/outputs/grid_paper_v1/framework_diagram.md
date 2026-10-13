# Framework diagram source

```mermaid
flowchart LR
    X[Public observation and base action] --> M[Legality mask and canonical action pool]
    M --> R[Replaceable ranking]
    R --> P[Initial 128 simulations]
    P --> D{Safe incumbent or exhausted?}
    D -- Yes --> O[Return verified proposal to base controller]
    D -- No --> B[Next batch of 64 simulations]
    B --> T{Safe incumbent or plateau after 256?}
    T -- No --> B
    T -- Yes --> L[Complete unqueried incumbent-substation candidates]
    L --> O
    B -- Pool exhausted --> O
    O --> C[Continuous control and restoration]
```

If no admissible incumbent exists, the extended search continues to exhaustion. Structural completion is conditional on extended-search stopping with an incumbent and remaining candidates. The diagram does not claim a hard runtime limit or future safety guarantee.

GNN is one possible ranker, not a necessary node in the main framework.
