# Momtrop

```momtrop``` is a rust library that implements the tropical Feynman samping algorithm for loop integrals in momentum space. It is designed with maximum flexibility in mind. The user has 
full control over over the evaluation of the integrand and the sampling is fully generic over floating point types. 


## Defining a Graph
To start integrating with ```momtrop``` one must first define a Graph. 
Graphs in ```momtrop``` are defined as a list of undirected edges. Each edge contains the 2 vertices that it connects, a boolean indicating whether it has a mass, and a ```f64``` specifying it's weight $\nu_e$
A graph also needs a list of the vertices that have an external momentum flowing into them. The following snippet defines a triangle graph:

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

## The SampleGenerator
Sampling is done from the ```SampleGenerator``` struct. A ```SampleGenerator``` can be build from a ```Graph``` by supplying a signature matrix. 
The following snippet builds a ```SampleGenerator``` for our triangle graph in $D = 3$ dimensions:

```rust
let loop_signature = vec![vec![1]; 3];
let sampler = graph.build_sampler::<3>(loop_signature).unwrap();
```

## Arbitrary Precision

If you want to run ```momtrop``` with floating point types other than f64, you must implement the ```MomtropFloat``` trait for your float of choice. 
For a full overview of the required methods, view the docs. If you want to implement ```MomtropFloat``` for a type that you do not own, you must 
wrap it in a new struct. 
