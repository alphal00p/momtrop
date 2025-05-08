# Momtrop

```momtrop``` is a rust library that implements the tropical Feynman samping algorithm for loop integrals in momentum space. It is designed with maximum flexibility in mind. The user has 
full control over over the evaluation of the integrand and the sampling is fully generic over floating point types. 


## Defining a Graph
To start integrating with ```momtrop``` one must first define a Graph. 
Graphs in ```momtrop``` are defined as a list of undirected edges. Each edge contains the 2 vertices that it connects, a boolean indicating whether it has a mass, and a ```f64``` specifying it's weight $\nu_e$.
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
Sampling is done from the ```SampleGenerator<D>``` struct, where ```D``` is the dimension. A ```SampleGenerator``` can be build from a ```Graph``` by supplying a signature matrix. 
The following snippet builds a ```SampleGenerator``` for our triangle graph in $D = 3$ dimensions:

```rust
let loop_signature = vec![vec![1]; 3];
let sampler = graph.build_sampler::<3>(loop_signature).unwrap();
```

## Kinematic data

In order to generate a sample point, one must supply the kinematic data associated to each edge. The vectors are supplied using ```momtrop```'s 
```Vector<T, D>``` type, where ```T``` is a floating point type. The kinematic data is supplied to the sampler using a ```Vec<(Option<T>, Vector<T, D>)>```. 
The ```Option<T>``` is the mass $m_e$ of an edge (```None``` should be used for massless edges). The vector provides the shift $p_e$ associated to an edge. 
For our triangle (assuming massless edges), the edge data may look like this:

```rust
use momtrop::Vector

let p1 = Vector::from_array([3.0, 4.0, 5.0]);
let p2 = Vector::from_array([6.0, 7.0, 8.0]);

let edge_data = vec![
        (None, Vector::new_from_num(&0.0)),
        (None, p1),
        (None, (&p1 + &p2)),
    ];

let settings = TropicalSamplingSettings {
        ..Default::default()
    };
```

## Generating a sample point 

    
```rust
use rand::SeedableRng;

let mut rng = rand::rngs::StdRng::seed_from_u64(69);

let settings = TropicalSamplingSettings {
        ..Default::default()
    };

let sample = sampler
            .generate_sample_from_rng(
                edge_data.clone(),
                &settings,
                &mut rng,
                #[cfg(feature = "log")]
                &logger,
            )
            .unwrap();
``` 
## Arbitrary Precision

If you want to run ```momtrop``` with floating point types other than f64, you must implement the ```MomtropFloat``` trait for your float of choice. 
For a full overview of the required methods, view the docs. If you want to implement ```MomtropFloat``` for a type that you do not own, you must 
wrap it in a new struct. 

## Reproducing the results from the paper 

The results can be reproduced using [```gammaloop```](https://github.com/alphal00p/gammaloop). The configuration files for each example 
are in the folder ```trop_paper_cards```. 

```
pip install gammaloop==0.3.3
gammaloop --build_dependencies
gammaloop trop_paper_cards/2_point_3_loop.gL
```
