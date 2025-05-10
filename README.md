# Momtrop

`momtrop` is a Rust library implementing the tropical Feynman sampling algorithm for loop integrals in momentum space (https://arxiv.org/abs/2504.09613). It is designed for maximum flexibility: the user retains full control over the evaluation of the integrand, and the sampling process is fully generic over floating-point types.

---

## Defining a Graph

To begin integrating with `momtrop`, you first need to define a graph. Graphs in `momtrop` are specified as a list of undirected edges. Each edge defines the two vertices it connects, a boolean indicating whether it has mass, and an `f64` value representing its weight ν_e.

You must also provide a list of vertices with incoming external momenta. The following example defines a triangle graph:

```rust
use momtrop::{Edge, Graph};

let weight = 2.0 / 3.0;

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

---

## The SampleGenerator

Sampling is performed using the `SampleGenerator<D>` struct, where `D` is the dimension. A `SampleGenerator` can be constructed from a `Graph` by supplying a signature matrix. The following code creates a `SampleGenerator` for our triangle graph in D = 3 dimensions:

```rust
let loop_signature = vec![vec![1]; 3];
let sampler = graph.build_sampler::<3>(loop_signature).unwrap();
```

---

## Kinematic Data

To generate a sample point, you must provide the kinematic data for each edge using the `Vector<T, D>` type, where `T` is a floating-point type.

This data is passed as a `Vec<(Option<T>, Vector<T, D>)>`. Each entry contains the optional mass m_e of the edge (use `None` for massless edges), and the shift vector p_e.

Example for massless edges:

```rust
use momtrop::Vector;

let p1 = Vector::from_array([3.0, 4.0, 5.0]);
let p2 = Vector::from_array([6.0, 7.0, 8.0]);

let edge_data = vec![
    (None, Vector::new_from_num(&0.0)),
    (None, p1),
    (None, &p1 + &p2),
];

let settings = TropicalSamplingSettings {
    ..Default::default()
};
```

---

## Generating a Sample Point

```rust
use rand::SeedableRng;

let mut rng = rand::rngs::StdRng::seed_from_u64(69);

let sample = sampler
    .generate_sample_from_rng(
        edge_data.clone(),
        &settings,
        &mut rng,
    )
    .unwrap();
```

Alternatively, you can supply the uniform random numbers in the hypercube manually:

```rust
let num_vars = sampler.get_dimension();
let x_space_point = repeat_with(|| rng.gen::<f64>()).take(num_vars).collect::<Vec<_>>();

let sample = sampler.generate_sample_from_x_space_point(
    x_space_point,
    edge_data,
    settings,
);
```

---

## Arbitrary Precision

To use `momtrop` with floating-point types other than `f64`, you must implement the `MomtropFloat` trait for your desired type. For a complete list of required methods, consult the documentation.

If you do not own the type (e.g., it is from an external crate), wrap it in a new struct before implementing the trait.

---

## Reproducing Results from the Paper

You can reproduce the results from the associated paper using [`gammaloop`](https://github.com/alphal00p/gammaloop). Configuration files for the examples are provided in the `trop_paper_cards` directory.

```bash
pip install gammaloop==0.3.3
gammaloop --build_dependencies
gammaloop trop_paper_cards/2_point_3_loop.gL
