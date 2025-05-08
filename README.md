## Defining a Graph

Graphs in ```momtrop``` are defined as a list of undirected edges. Each edge contains the 2 vertices that it connects, a boolean indicating whether it has a mass, and a ```f64``` specifying it's weight $\nu_e$
A graph also needs a list of the vertices that have an external momentum flowing into them. 

```rust
 use momtrop::{Edge, Graph}

 let weight = 2. / 3.;

    let triangle_edges = vec![
        Edge {
            vertices: (0, 1),
            is_massive: false,
            weight,
        },
        Edge {
            vertices: (1, 2),
            is_massive: false,
            weight,
        },
        Edge {
            vertices: (2, 0),
            is_massive: false,
            weight,
        },
    ];

    let externals = vec![0, 1, 2];

    let graph = Graph {
        edges: triangle_edges,
        externals,
    };
```

## Arbitrary Precision

If you want to run ```momtrop``` with floating point types other than f64, you must implement the ```MomtropFloat``` trait for your float of choice. 
For a full overview of the required methods, view the docs. 
