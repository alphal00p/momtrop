use std::iter::repeat_with;

#[cfg(feature = "log")]
use log::Logger;

use f128::f128;
use float::FloatLike;
use itertools::Itertools;
use preprocessing::{TropicalGraph, TropicalSubgraphTable};
use rand::Rng;
use sampling::{sample, SamplingError};
use serde::{Deserialize, Serialize};
use vector::Vector;

mod float;
pub mod gamma;
#[cfg(feature = "log")]
pub mod log;
mod matrix;
mod mimic_rng;
mod preprocessing;
mod sampling;
#[cfg(feature = "sympol")]
mod symbolic_polynomail;
pub mod vector;

pub const MAX_EDGES: usize = 64;
pub const MAX_VERTICES: usize = 256;

#[derive(Debug)]
pub struct TropicalSamplingSettings {
    pub upcast_on_failure: bool,
    pub matrix_stability_test: Option<f64>,
    pub print_debug_info: bool,
}

impl Default for TropicalSamplingSettings {
    fn default() -> Self {
        Self {
            upcast_on_failure: true,
            matrix_stability_test: None,
            print_debug_info: false,
        }
    }
}

#[cfg(test)]
fn assert_approx_eq(res: f64, target: f64, tolerance: f64) {
    if approx_eq(res, target, tolerance) {
    } else {
        panic!(
            "assert_approx_eq failed: \n{:+e} != \n{:+e} with tolerance {:+e}",
            res, target, tolerance
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
pub struct TropicalSampleResult<T: FloatLike, const D: usize> {
    pub loop_momenta: Vec<Vector<T, D>>,
    pub u_trop: T,
    pub v_trop: T,
    pub u: T,
    pub v: T,
    pub jacobian: T,
}

impl<const D: usize> TropicalSampleResult<f128, D> {
    pub fn downcast(&self) -> TropicalSampleResult<f64, D> {
        TropicalSampleResult {
            loop_momenta: self.loop_momenta.iter().map(|v| v.downcast()).collect_vec(),
            u_trop: self.u_trop.into(),
            v_trop: self.v_trop.into(),
            u: self.u.into(),
            v: self.v.into(),
            jacobian: self.jacobian.into(),
        }
    }
}

impl Graph {
    pub fn build_sampler<const D: usize>(
        self,
        loop_signature: Vec<Vec<isize>>,
        dimension: usize,
    ) -> Result<SampleGenerator<D>, String> {
        let tropical_graph = TropicalGraph::from_graph(self, dimension);
        let table = TropicalSubgraphTable::generate_from_tropical(&tropical_graph, dimension)?;

        Ok(SampleGenerator {
            loop_signature,
            table,
        })
    }
}

#[cfg(test)]
fn approx_eq(res: f64, target: f64, tolerance: f64) -> bool {
    if target == 0.0 {
        res.abs() < tolerance
    } else {
        ((res - target) / target).abs() < tolerance
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleGenerator<const D: usize> {
    loop_signature: Vec<Vec<isize>>,
    table: TropicalSubgraphTable,
}

impl<const D: usize> SampleGenerator<D> {
    pub fn generate_sample_from_x_space_point<#[cfg(feature = "log")] L: Logger>(
        &self,
        x_space_point: &[f64],
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        settings: &TropicalSamplingSettings,
        #[cfg(feature = "log")] logger: &L,
    ) -> Result<TropicalSampleResult<f64, D>, SamplingError> {
        let sample = sample(
            &self.table,
            x_space_point,
            &self.loop_signature,
            &edge_data,
            settings,
            #[cfg(feature = "log")]
            logger,
        );

        match sample {
            Ok(sample) => Ok(sample),
            Err(sampling_error) => {
                if settings.upcast_on_failure {
                    match sampling_error {
                        SamplingError::MatrixError(_matrix_error) => {
                            let res = self.generate_sample_f128_from_x_space_point(
                                x_space_point,
                                edge_data,
                                settings,
                                #[cfg(feature = "log")]
                                logger,
                            );

                            res.map(|res| res.downcast())
                        }
                    }
                } else {
                    Err(sampling_error)
                }
            }
        }
    }

    pub fn generate_sample_from_rng<R: Rng, #[cfg(feature = "log")] L: Logger>(
        &self,
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        settings: &TropicalSamplingSettings,
        rng: &mut R,
        #[cfg(feature = "log")] logger: &L,
    ) -> Result<TropicalSampleResult<f64, D>, SamplingError> {
        let num_vars = self.get_dimension();
        let x_space_point = repeat_with(|| rng.gen::<f64>())
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

    pub fn generate_sample_f128_from_x_space_point<#[cfg(feature = "log")] L: Logger>(
        &self,
        x_space_point: &[f64],
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        settings: &TropicalSamplingSettings,
        #[cfg(feature = "log")] logger: &L,
    ) -> Result<TropicalSampleResult<f128, D>, SamplingError> {
        let upcasted_xspace_point = x_space_point.iter().copied().map(f128::new).collect_vec();
        let upcasted_edge_data = edge_data
            .into_iter()
            .map(|(mass, edge_shift)| (mass.map(f128::new), edge_shift.upcast()))
            .collect_vec();

        sample(
            &self.table,
            &upcasted_xspace_point,
            &self.loop_signature,
            &upcasted_edge_data,
            settings,
            #[cfg(feature = "log")]
            logger,
        )
    }

    pub fn generate_sample_f128_from_rng<R: Rng, #[cfg(feature = "log")] L: Logger>(
        &self,
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        settings: &TropicalSamplingSettings,
        rng: &mut R,
        #[cfg(feature = "log")] logger: &L,
    ) -> Result<TropicalSampleResult<f128, D>, SamplingError> {
        let num_vars = self.get_dimension();
        let x_space_point = repeat_with(|| rng.gen::<f64>())
            .take(num_vars)
            .collect_vec();

        self.generate_sample_f128_from_x_space_point(
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
