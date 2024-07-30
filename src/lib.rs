use std::iter::repeat_with;

use f128::f128;
use float::FloatLike;
use itertools::Itertools;
use preprocessing::{TropicalGraph, TropicalSubgraphTable};
use rand::Rng;
use sampling::sample;
use serde::{Deserialize, Serialize};
use vector::Vector;

mod float;
pub mod gamma;
mod matrix;
mod mimic_rng;
mod preprocessing;
mod sampling;
pub mod vector;

pub const MAX_EDGES: usize = 64;
pub const MAX_VERTICES: usize = 256;

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

pub struct TropicalSampleResult<T: FloatLike, const D: usize> {
    pub loop_momenta: Vec<Vector<T, D>>,
    pub u_trop: T,
    pub v_trop: T,
    pub u: T,
    pub v: T,
    pub prefactor: T,
    pub jacobian: T,
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
    pub fn generate_sample_from_x_space_point(
        &self,
        x_space_point: &[f64],
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        print_debug_info: bool,
    ) -> TropicalSampleResult<f64, D> {
        sample::<f64, D>(
            &self.table,
            x_space_point,
            &self.loop_signature,
            edge_data,
            print_debug_info,
        )
    }

    pub fn generate_sample_from_rng<R: Rng>(
        &self,
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        print_debug_info: bool,
        rng: &mut R,
    ) -> TropicalSampleResult<f64, D> {
        let num_vars = self.get_dimension();
        let x_space_point = repeat_with(|| rng.gen::<f64>())
            .take(num_vars)
            .collect_vec();

        self.generate_sample_from_x_space_point(&x_space_point, edge_data, print_debug_info)
    }

    pub fn generate_sample_f128_from_x_space_point(
        &self,
        x_space_point: &[f64],
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        print_debug_info: bool,
    ) -> TropicalSampleResult<f128, D> {
        let upcasted_xspace_point = x_space_point.iter().copied().map(f128::new).collect_vec();
        let upcasted_edge_data = edge_data
            .into_iter()
            .map(|(mass, edge_shift)| (mass.map(f128::new), edge_shift.upcast()))
            .collect();

        sample::<f128, D>(
            &self.table,
            &upcasted_xspace_point,
            &self.loop_signature,
            upcasted_edge_data,
            print_debug_info,
        )
    }

    pub fn generate_sample_f128_from_rng<R: Rng>(
        &self,
        edge_data: Vec<(Option<f64>, vector::Vector<f64, D>)>,
        print_debug_info: bool,
        rng: &mut R,
    ) -> TropicalSampleResult<f128, D> {
        let num_vars = self.get_dimension();
        let x_space_point = repeat_with(|| rng.gen::<f64>())
            .take(num_vars)
            .collect_vec();

        self.generate_sample_f128_from_x_space_point(&x_space_point, edge_data, print_debug_info)
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
}
