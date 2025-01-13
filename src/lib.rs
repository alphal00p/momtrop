use std::iter::repeat_with;

#[cfg(feature = "log")]
use log::Logger;

use float::MomTropFloat;
use itertools::Itertools;
use matrix::{DecompositionResult, SquareMatrix};
use preprocessing::{TropicalGraph, TropicalSubgraphTable};
use rand::Rng;
use sampling::{sample, SamplingError};
use serde::{Deserialize, Serialize};
use vector::Vector;

pub mod float;
pub mod gamma;
#[cfg(feature = "log")]
pub mod log;
pub mod matrix;
mod mimic_rng;
mod preprocessing;
mod sampling;
pub mod vector;

pub const MAX_EDGES: usize = 64;
pub const MAX_VERTICES: usize = 256;

#[derive(Debug)]
pub struct TropicalSamplingSettings {
    pub upcast_on_failure: bool,
    pub matrix_stability_test: Option<f64>,
    pub print_debug_info: bool,
    pub return_metadata: bool,
}

impl Default for TropicalSamplingSettings {
    fn default() -> Self {
        Self {
            upcast_on_failure: true,
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

pub struct Graph {
    pub edges: Vec<Edge>,
    pub externals: Vec<u8>,
}

pub struct Edge {
    pub vertices: (u8, u8),
    pub is_massive: bool,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct TropicalSampleResult<T: MomTropFloat, const D: usize> {
    pub loop_momenta: Vec<Vector<T, D>>,
    pub u_trop: T,
    pub v_trop: T,
    pub u: T,
    pub v: T,
    pub jacobian: T,
    pub metadata: Option<Metadata<T, D>>,
}

#[derive(Debug, Clone)]
pub struct Metadata<T: MomTropFloat, const D: usize> {
    pub q_vectors: Vec<Vector<T, D>>,
    pub lambda: T,
    pub l_matrix: SquareMatrix<T>,
    pub decompoisiton_result: DecompositionResult<T>,
    pub u_vectors: Vec<Vector<T, D>>,
    pub shift: Vec<Vector<T, D>>,
}

impl Graph {
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
pub struct SampleGenerator<const D: usize> {
    loop_signature: Vec<Vec<isize>>,
    table: TropicalSubgraphTable,
}

impl<const D: usize> SampleGenerator<D> {
    pub fn generate_sample_from_x_space_point<
        T: MomTropFloat,
        #[cfg(feature = "log")] L: Logger,
    >(
        &self,
        x_space_point: &[T],
        edge_data: Vec<(Option<T>, vector::Vector<T, D>)>,
        settings: &TropicalSamplingSettings,
        #[cfg(feature = "log")] logger: &L,
    ) -> Result<TropicalSampleResult<T, D>, SamplingError> {
        sample(
            &self.table,
            x_space_point,
            &self.loop_signature,
            &edge_data,
            settings,
            #[cfg(feature = "log")]
            logger,
        )
    }

    pub fn generate_sample_from_rng<T: MomTropFloat, R: Rng, #[cfg(feature = "log")] L: Logger>(
        &self,
        edge_data: Vec<(Option<T>, vector::Vector<T, D>)>,
        settings: &TropicalSamplingSettings,
        rng: &mut R,
        #[cfg(feature = "log")] logger: &L,
    ) -> Result<TropicalSampleResult<T, D>, SamplingError> {
        let const_builder = edge_data[0].1.zero();

        let num_vars = self.get_dimension();
        let x_space_point = repeat_with(|| const_builder.from_f64(rng.gen::<f64>()))
            .take(num_vars)
            .collect_vec();

        self.generate_sample_from_x_space_point(
            &x_space_point,
            edge_data,
            settings,
            #[cfg(feature = "log")]
            logger,
        )
    }

    pub fn get_dimension(&self) -> usize {
        self.table.get_num_variables()
    }

    pub fn get_dod(&self) -> f64 {
        self.table.tropical_graph.dod
    }

    pub fn iter_edge_weights(&self) -> impl Iterator<Item = f64> + '_ {
        self.table.tropical_graph.topology.iter().map(|e| e.weight)
    }

    pub fn get_num_edges(&self) -> usize {
        self.table.tropical_graph.topology.len()
    }

    pub fn get_smallest_dod(&self) -> f64 {
        self.table.get_smallest_dod()
    }
}
