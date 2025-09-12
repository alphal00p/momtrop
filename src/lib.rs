//! `momtrop` is a flexible Rust library for tropical Feynman sampling of loop integrals in momentum space (https://arxiv.org/abs/2504.09613).
//! It allows users to define graphs with arbitrary edge weights, dimensions, and external kinematics,
//! and supports custom floating-point types via the `MomtropFloat` trait.
//!
//! The library only takes care of sample generation, and defers the evaluation of the integrand to the user.

use std::iter::repeat_with;

use bincode::{Decode, Encode};

use float::MomTropFloat;
use itertools::Itertools;
use matrix::{DecompositionResult, SquareMatrix};
use preprocessing::{TropicalGraph, TropicalSubgraphTable};
use rand::Rng;
use sampling::{SamplingError, sample};
use serde::{Deserialize, Serialize};
use vector::Vector;

pub mod float;
pub mod gamma;
pub mod matrix;
mod mimic_rng;
mod preprocessing;
mod sampling;
pub mod vector;

/// Maximum number of edges supported by momtrop.
pub const MAX_EDGES: usize = 64;
/// Maximum number of vertices supported by momtrop.
pub const MAX_VERTICES: usize = 256;

#[derive(Debug)]
/// Struct containing all runtime settings.
pub struct TropicalSamplingSettings {
    /// `matrix_stability_test` tests the numerical stability
    /// of some of the matrix routines used during sampling. This is done by checking
    /// how far L L^-1 is from the identity matrix in terms of a L_2_1 norm. If this distance is
    /// larger than the float provided, the sampling will return an Err.
    pub matrix_stability_test: Option<f64>,
    /// Setting `print_debug_info` to `true` will print out the results of intermediate steps during sampling.
    /// Useful for debugging.
    pub print_debug_info: bool,
    /// Enabling `return_metadata` provides access to some of the intermediate results. This is useful for
    /// exploring new applications.
    pub return_metadata: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for TropicalSamplingSettings {
    fn default() -> Self {
        Self {
            matrix_stability_test: None,
            print_debug_info: false,
            return_metadata: false,
        }
    }
}

pub fn assert_approx_eq<T: MomTropFloat>(res: &T, target: &T, tolerance: &T) {
    if approx_eq(res, target, tolerance) {
    } else {
        panic!(
            "assert_approx_eq failed: \n{:?} != \n{:?} with tolerance {:?}",
            &res, &target, &tolerance
        )
    }
}

/// Main graph struct from which a sampler can be build. This graph should be stripped of
/// tree-like and external edges.
/// Vertices which would have external edges attached to them must be added to the `externals` field.
///
pub struct Graph {
    pub edges: Vec<Edge>,
    pub externals: Vec<u8>,
}

/// Edge struct from which a graph can be specified.
pub struct Edge {
    pub vertices: (u8, u8),
    pub is_massive: bool,
    /// `weight` specifies the power of the propagator corresponding to an edge.
    pub weight: f64,
}

#[derive(Debug, Clone)]
/// Return type of the sample function
pub struct TropicalSampleResult<T: MomTropFloat, const D: usize> {
    /// loop momenta of sample point.
    pub loop_momenta: Vec<Vector<T, D>>,
    /// tropical approximation of U at sample point.
    pub u_trop: T,
    /// tropical approximation of V at sample point.
    pub v_trop: T,
    /// U at sample point.
    pub u: T,
    /// V at sample point.
    pub v: T,
    /// Jacobian at sample point.
    pub jacobian: T,
    /// metadata, is `None` is `return_metadata` is disabled.
    pub metadata: Option<Metadata<T, D>>,
}

#[derive(Debug, Clone)]
/// Optional metadata for advanced users, see \citation for the definition of the fields
/// in this struct.
pub struct Metadata<T: MomTropFloat, const D: usize> {
    /// Vectors q sampled from Gaussian.
    pub q_vectors: Vec<Vector<T, D>>,
    /// Parameter lambda sampled from Gamma distribution.
    pub lambda: T,
    /// Matrix L at sample point.
    pub l_matrix: SquareMatrix<T>,
    /// Cholesky decomposition and inverse of L at sample point.
    pub decompoisiton_result: DecompositionResult<T>,
    /// The vectors u at sample point.
    pub u_vectors: Vec<Vector<T, D>>,
    /// L^-1 u at sample point
    pub shift: Vec<Vector<T, D>>,
}

impl Graph {
    /// Build the sampler from a graph. `loop_signature` is the signature matrix determining the loop-momentum routing.
    /// Returns `Err` if a divergent subgraph is found.
    ///
    /// For example, the following code builds a sampler for the massless triangle with all edges having weight 2/3.
    ///
    ///```rust
    ///use momtrop::{Graph, Edge};
    ///
    ///let weight = 2.0 / 3.0;
    ///
    ///let triangle_edges = vec![
    ///    Edge {
    ///        vertices: (0, 1),
    ///        is_massive: false,
    ///        weight,
    ///    },
    ///    Edge {
    ///        vertices: (1, 2),
    ///        is_massive: false,
    ///        weight,
    ///    },
    ///    Edge {
    ///        vertices: (2, 0),
    ///        is_massive: false,
    ///        weight,
    ///    },
    ///];
    ///
    ///let externals = vec![0, 1, 2];
    ///
    ///let triangle_graph = Graph {
    ///    edges: triangle_edges,
    ///    externals,
    ///};
    ///
    /// let loop_signature = vec![vec![1]; 3];
    /// let triangle_sampler = triangle_graph.build_sampler::<3>(loop_signature);
    ///
    /// assert!(triangle_sampler.is_ok());
    /// ```
    pub fn build_sampler<const D: usize>(
        self,
        loop_signature: Vec<Vec<isize>>,
    ) -> Result<SampleGenerator<D>, String> {
        let tropical_graph = TropicalGraph::from_graph(self, D);
        let table = TropicalSubgraphTable::generate_from_tropical(&tropical_graph, D)?;

        Ok(SampleGenerator {
            loop_signature,
            table,
        })
    }
}

fn approx_eq<T: MomTropFloat>(res: &T, target: &T, tolerance: &T) -> bool {
    if target == &res.zero() {
        &res.abs() < tolerance
    } else {
        &((res.ref_sub(target)) / target).abs() < tolerance
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Decode, Encode)]
/// Sampler struct from which sample points can be generated.
pub struct SampleGenerator<const D: usize> {
    loop_signature: Vec<Vec<isize>>,
    table: TropicalSubgraphTable,
}

impl<const D: usize> SampleGenerator<D> {
    /// Generate a sample point for a given random point `x_space_point` in the unit hypercube.
    /// `edge_data` contains masses and shifs of each propagator.
    /// Dimensionality of the unit hypercube, should match the length of `x_space_point`.
    ///```rust
    ///# use momtrop::{Graph, Edge};
    ///use momtrop::{TropicalSamplingSettings, vector::Vector};
    ///use rand::{rngs::StdRng, SeedableRng, Rng};
    ///use std::iter::repeat_with;
    ///#
    ///# let weight = 2.0 / 3.0;
    ///#
    ///# let triangle_edges = vec![
    ///#     Edge {
    ///#         vertices: (0, 1),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (1, 2),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (2, 0),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///# ];
    ///#
    ///# let externals = vec![0, 1, 2];
    ///#
    ///# let triangle_graph = Graph {
    ///#     edges: triangle_edges,
    ///#     externals,
    ///# };
    ///#
    ///#  let loop_signature = vec![vec![1]; 3];
    ///let triangle_sampler = triangle_graph.build_sampler::<3>(loop_signature).unwrap();
    ///
    ///let settings = TropicalSamplingSettings::default();
    ///
    ///let p_1 = Vector::from_array([0.1, 0.2, 0.3]);
    ///let p_2 = Vector::from_array([0.4, 0.5, 0.6]);
    ///let edge_data = vec![(None, Vector::new_from_num(&0.0)), (None, p_1), (None, p_2)];
    ///
    ///let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    ///
    ///let x_space_point = repeat_with(|| rng.r#gen::<f64>()).take(triangle_sampler.get_dimension()).collect::<Vec<_>>();
    ///let sample = triangle_sampler.generate_sample_from_x_space_point(&x_space_point, edge_data, &settings);
    /// assert!(sample.is_ok());
    ///
    /// ```
    pub fn generate_sample_from_x_space_point<T: MomTropFloat>(
        &self,
        x_space_point: &[T],
        edge_data: Vec<(Option<T>, vector::Vector<T, D>)>,
        settings: &TropicalSamplingSettings,
    ) -> Result<TropicalSampleResult<T, D>, SamplingError> {
        sample(
            &self.table,
            x_space_point,
            &self.loop_signature,
            &edge_data,
            settings,
        )
    }

    /// Alternative to `generate_sample_from_x_space_point`. Uses a `Rng` to generate the point
    /// in the unit hypercube.
    ///```rust
    ///# use momtrop::{Graph, Edge};
    ///use momtrop::{TropicalSamplingSettings, vector::Vector};
    ///use rand::{rngs::StdRng, SeedableRng, Rng};
    ///use std::iter::repeat_with;
    ///#
    ///# let weight = 2.0 / 3.0;
    ///#
    ///# let triangle_edges = vec![
    ///#     Edge {
    ///#         vertices: (0, 1),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (1, 2),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (2, 0),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///# ];
    ///#
    ///# let externals = vec![0, 1, 2];
    ///#
    ///# let triangle_graph = Graph {
    ///#     edges: triangle_edges,
    ///#     externals,
    ///# };
    ///#
    ///#  let loop_signature = vec![vec![1]; 3];
    ///let triangle_sampler = triangle_graph.build_sampler::<3>(loop_signature).unwrap();
    ///
    ///let settings = TropicalSamplingSettings::default();
    ///
    ///let p_1 = Vector::from_array([0.1, 0.2, 0.3]);
    ///let p_2 = Vector::from_array([0.4, 0.5, 0.6]);
    ///let edge_data = vec![(None, Vector::new_from_num(&0.0)), (None, p_1), (None, p_2)];
    ///
    ///let mut rng: StdRng = SeedableRng::seed_from_u64(42);
    ///
    ///let sample = triangle_sampler.generate_sample_from_rng(edge_data, &settings, &mut rng);
    /// assert!(sample.is_ok());
    ///
    /// ```
    pub fn generate_sample_from_rng<T: MomTropFloat, R: Rng>(
        &self,
        edge_data: Vec<(Option<T>, vector::Vector<T, D>)>,
        settings: &TropicalSamplingSettings,
        rng: &mut R,
    ) -> Result<TropicalSampleResult<T, D>, SamplingError> {
        let const_builder = edge_data[0].1.zero();

        let num_vars = self.get_dimension();
        let x_space_point = repeat_with(|| const_builder.from_f64(rng.r#gen::<f64>()))
            .take(num_vars)
            .collect_vec();

        self.generate_sample_from_x_space_point(&x_space_point, edge_data, settings)
    }

    /// Dimensionality of the unit hypercube, should match the length of `x_space_point`.
    ///```rust
    ///# use momtrop::{Graph, Edge};
    ///#
    ///# let weight = 2.0 / 3.0;
    ///#
    ///# let triangle_edges = vec![
    ///#     Edge {
    ///#         vertices: (0, 1),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (1, 2),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (2, 0),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///# ];
    ///#
    ///# let externals = vec![0, 1, 2];
    ///#
    ///# let triangle_graph = Graph {
    ///#     edges: triangle_edges,
    ///#     externals,
    ///# };
    ///#
    ///#  let loop_signature = vec![vec![1]; 3];
    ///let triangle_sampler = triangle_graph.build_sampler::<3>(loop_signature).unwrap();
    ///
    /// assert_eq!(triangle_sampler.get_dimension(), 2*3 - 1 + 3 + (3 % 2) );
    /// ```
    pub fn get_dimension(&self) -> usize {
        self.table.get_num_variables()
    }

    /// Degree of divergence of the provided graph.
    ///```rust
    ///# use momtrop::{Graph, Edge};
    ///#
    ///# let weight = 2.0 / 3.0;
    ///#
    ///# let triangle_edges = vec![
    ///#     Edge {
    ///#         vertices: (0, 1),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (1, 2),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (2, 0),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///# ];
    ///#
    ///# let externals = vec![0, 1, 2];
    ///#
    ///# let triangle_graph = Graph {
    ///#     edges: triangle_edges,
    ///#     externals,
    ///# };
    ///#
    ///#  let loop_signature = vec![vec![1]; 3];
    ///let triangle_sampler = triangle_graph.build_sampler::<3>(loop_signature).unwrap();
    ///
    /// assert_eq!(triangle_sampler.get_dod(), 3.0 * 2.0 / 3.0 - 3.0 / 2.0);
    /// ```
    pub fn get_dod(&self) -> f64 {
        self.table.tropical_graph.dod
    }

    /// Iterate over the edge weights of the provided graph.
    ///```rust
    ///# use momtrop::{Graph, Edge};
    ///#
    ///# let weight = 2.0 / 3.0;
    ///#
    ///# let triangle_edges = vec![
    ///#     Edge {
    ///#         vertices: (0, 1),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (1, 2),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (2, 0),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///# ];
    ///#
    ///# let externals = vec![0, 1, 2];
    ///#
    ///# let triangle_graph = Graph {
    ///#     edges: triangle_edges,
    ///#     externals,
    ///# };
    ///#
    ///#  let loop_signature = vec![vec![1]; 3];
    ///let triangle_sampler = triangle_graph.build_sampler::<3>(loop_signature).unwrap();
    ///
    /// assert!(triangle_sampler.iter_edge_weights().all(|w| w == 2.0/3.0));
    /// ```
    pub fn iter_edge_weights(&self) -> impl Iterator<Item = f64> + '_ {
        self.table.tropical_graph.topology.iter().map(|e| e.weight)
    }

    /// Get the number of edges of the provided graph.
    ///```rust
    ///# use momtrop::{Graph, Edge};
    ///#
    ///# let weight = 2.0 / 3.0;
    ///#
    ///# let triangle_edges = vec![
    ///#     Edge {
    ///#         vertices: (0, 1),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (1, 2),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///#     Edge {
    ///#         vertices: (2, 0),
    ///#         is_massive: false,
    ///#         weight,
    ///#     },
    ///# ];
    ///#
    ///# let externals = vec![0, 1, 2];
    ///#
    ///# let triangle_graph = Graph {
    ///#     edges: triangle_edges,
    ///#     externals,
    ///# };
    ///#
    ///#  let loop_signature = vec![vec![1]; 3];
    ///let triangle_sampler = triangle_graph.build_sampler::<3>(loop_signature).unwrap();
    ///
    /// assert_eq!(triangle_sampler.get_num_edges(), 3);
    /// ```
    pub fn get_num_edges(&self) -> usize {
        self.table.tropical_graph.topology.len()
    }

    /// Get the smallest degree of divergence in the subgraph table.
    pub fn get_smallest_dod(&self) -> f64 {
        self.table.get_smallest_dod()
    }
}
