use float::MomTropFloat;
use itertools::Itertools;
use log::Logger;
use matrix::{DecompositionResult, SquareMatrix};
use preprocessing::{TropicalGraph, TropicalSubgraphTable};
use rand::Rng;
use sampling::{sample, SamplingError};
use serde::{Deserialize, Serialize};
use std::iter::repeat_with;
use vector::Vector;

pub mod float;
pub mod gamma;
pub mod log;
pub mod matrix;
mod mimic_rng;
mod preprocessing;
mod sampling;
pub mod vector;

#[cfg(feature = "python_api")]
pub mod python;

/// Maximum number of edges supported by momtrop.
pub const MAX_EDGES: usize = 64;
/// Maximum number of vertices supported by momtrop.
pub const MAX_VERTICES: usize = 256;

#[derive(Debug, Clone)]
/// Struct containing all runtime settings.
pub struct TropicalSamplingSettings<L: Logger = ()> {
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
    /// Optional logger for logging messages during sampling.
    pub logger: Option<L>,
}

#[allow(clippy::derivable_impls)]
impl<L: Logger> Default for TropicalSamplingSettings<L> {
    fn default() -> Self {
        Self {
            matrix_stability_test: None,
            print_debug_info: false,
            return_metadata: false,
            logger: None,
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
#[derive(Clone)]
pub struct Graph {
    pub edges: Vec<Edge>,
    pub externals: Vec<u8>,
}

/// Edge struct from which a graph can be specified.
#[derive(Clone, Copy)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Sampler struct from which sample points can be generated.
pub struct SampleGenerator<const D: usize> {
    loop_signature: Vec<Vec<isize>>,
    table: TropicalSubgraphTable,
}

impl<const D: usize> SampleGenerator<D> {
    /// Generate a sample point for a given random point `x_space_point` in the unit hypercube.
    /// `edge_data` contains masses and shifs of each propagator.
    pub fn generate_sample_from_x_space_point<T: MomTropFloat, L: Logger>(
        &self,
        x_space_point: &[T],
        edge_data: Vec<(Option<T>, vector::Vector<T, D>)>,
        settings: &TropicalSamplingSettings<L>,
        force_sector: Option<&[usize]>,
    ) -> Result<TropicalSampleResult<T, D>, SamplingError> {
        sample(
            &self.table,
            x_space_point,
            &self.loop_signature,
            &edge_data,
            settings,
            force_sector,
        )
    }

    /// Alternative to `generate_sample_from_x_space_point`. Uses a `Rng` to generate the point
    /// in the unit hypercube.
    pub fn generate_sample_from_rng<T: MomTropFloat, R: Rng, L: Logger>(
        &self,
        edge_data: Vec<(Option<T>, vector::Vector<T, D>)>,
        settings: &TropicalSamplingSettings<L>,
        rng: &mut R,
    ) -> Result<TropicalSampleResult<T, D>, SamplingError> {
        let const_builder = edge_data[0].1.zero();

        let num_vars = self.get_dimension();
        let x_space_point = repeat_with(|| const_builder.from_f64(rng.gen::<f64>()))
            .take(num_vars)
            .collect_vec();

        self.generate_sample_from_x_space_point(&x_space_point, edge_data, settings, None)
    }

    /// Dimensionality of the unit hypercube, should match the length of `x_space_point`.
    pub fn get_dimension(&self) -> usize {
        self.table.get_num_variables()
    }

    /// Degree of divergence of the provided graph.
    pub fn get_dod(&self) -> f64 {
        self.table.tropical_graph.dod
    }

    /// Iterate over the edge weights of the provided graph.
    pub fn iter_edge_weights(&self) -> impl Iterator<Item = f64> + '_ {
        self.table.tropical_graph.topology.iter().map(|e| e.weight)
    }

    /// Get the number of edges of the provided graph.
    pub fn get_num_edges(&self) -> usize {
        self.table.tropical_graph.topology.len()
    }

    /// Get the smallest degree of divergence in the subgraph table.
    pub fn get_smallest_dod(&self) -> f64 {
        self.table.get_smallest_dod()
    }
}
