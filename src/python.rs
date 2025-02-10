use itertools::Itertools;
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods, pymodule,
    types::{PyModule, PyModuleMethods},
    Bound, IntoPyObject, PyResult,
};

use crate::{
    vector::Vector, Edge, Graph, SampleGenerator, TropicalSampleResult, TropicalSamplingSettings,
};

#[pyclass(name = "Edge")]
#[derive(Clone)]
pub struct PythonEdge {
    edge: Edge,
}

#[pymethods]
impl PythonEdge {
    #[new]
    fn new(vertices: (u8, u8), is_massive: bool, weight: f64) -> Self {
        PythonEdge {
            edge: Edge {
                vertices,
                is_massive,
                weight,
            },
        }
    }
}

#[pyclass(name = "Graph")]
#[derive(Clone)]
pub struct PythonGraph {
    graph: Graph,
}

#[pymethods]
impl PythonGraph {
    #[new]
    fn new(edges: Vec<PythonEdge>, externals: Vec<u8>) -> Self {
        let rust_edges = edges
            .into_iter()
            .map(|python_edge| python_edge.edge)
            .collect_vec();

        Self {
            graph: Graph {
                edges: rust_edges,
                externals,
            },
        }
    }
}

#[pyclass(name = "Sampler")]
pub struct PythonSampler {
    sampler: SampleGenerator<3>,
}

#[pymethods]
impl PythonSampler {
    #[new]
    /// build a new sampler from a graph and associated signature
    fn new(graph: PythonGraph, loop_signature: Vec<Vec<isize>>) -> PyResult<Self> {
        match graph.graph.build_sampler(loop_signature) {
            Ok(sampler) => Ok(PythonSampler { sampler }),
            Err(error_message) => Err(PyValueError::new_err(error_message)),
        }
    }

    /// Get the dimensionality of the unit hypercube
    pub fn get_dimension(&self) -> usize {
        self.sampler.get_dimension()
    }

    /// Get the number of edges in the graph
    pub fn get_num_edges(&self) -> usize {
        self.sampler.get_num_edges()
    }

    #[pyo3(signature = (x_space_point, edge_data, settings, force_sector=None))]
    pub fn sample_point(
        &self,
        x_space_point: Vec<f64>,
        edge_data: PythonEdgeData,
        settings: PythonSettings,
        force_sector: Option<Vec<usize>>,
    ) -> PyResult<PythonTropicalSampleResult> {
        let rust_result = self.sampler.generate_sample_from_x_space_point(
            &x_space_point,
            edge_data.data,
            &settings.settings,
            force_sector.as_deref(),
        )?;

        let python_result = PythonTropicalSampleResult {
            result: rust_result,
        };

        Ok(python_result)
    }

    /// provides the probability of each edge in the same order as they are supplied
    pub fn get_subgraph_pdf(&self, subgraph: Vec<usize>) -> Vec<f64> {
        self.sampler.table.get_subgraph_pdf(&subgraph)
    }

    pub fn get_sector_prob(&self, sector: Vec<usize>) -> f64 {
        self.sampler.table.get_sector_prob(&sector)
    }

    pub fn get_itr(&self) -> f64 {
        self.sampler.table.table.last().unwrap().j_function
    }
}

#[pyclass(name = "Settings")]
#[derive(Clone)]
pub struct PythonSettings {
    settings: TropicalSamplingSettings,
}

#[pymethods]
impl PythonSettings {
    #[new]
    #[pyo3(signature = (print_debug_info, return_metadata, matrix_stability_test=None))]
    fn new(
        print_debug_info: bool,
        return_metadata: bool,
        matrix_stability_test: Option<f64>,
    ) -> Self {
        Self {
            settings: TropicalSamplingSettings {
                matrix_stability_test,
                print_debug_info,
                return_metadata,
                logger: None,
            },
        }
    }
}

#[pyclass(name = "Vector")]
#[derive(Clone)]
pub struct PythonVector {
    vector: Vector<f64, 3>,
}

#[pymethods]
impl PythonVector {
    #[new]
    fn new(x: f64, y: f64, z: f64) -> Self {
        PythonVector {
            vector: Vector::from_array([x, y, z]),
        }
    }

    fn __repr__(&self) -> PyResult<impl IntoPyObject> {
        Ok(format!(
            "x: {}, y: {}, z: {}",
            self.vector[0], self.vector[1], self.vector[2]
        ))
    }
}

#[pyclass(name = "TropicalSampleResult")]
pub struct PythonTropicalSampleResult {
    result: TropicalSampleResult<f64, 3>,
}

#[pymethods]
impl PythonTropicalSampleResult {
    #[getter]
    fn get_loop_momenta(&self) -> Vec<PythonVector> {
        self.result
            .loop_momenta
            .iter()
            .map(|&loop_momentum| PythonVector {
                vector: loop_momentum,
            })
            .collect()
    }

    #[getter]
    fn get_jacobian(&self) -> f64 {
        self.result.jacobian
    }
}

#[derive(Clone)]
#[pyclass(name = "EdgeData")]
pub struct PythonEdgeData {
    data: Vec<(Option<f64>, Vector<f64, 3>)>,
}

#[pymethods]
impl PythonEdgeData {
    #[new]
    fn new(masses: Vec<f64>, external_shifts: Vec<PythonVector>) -> PyResult<Self> {
        if masses.len() != external_shifts.len() {
            return Err(PyValueError::new_err(
                "mass vector and shifts vector of unequal lengt",
            ));
        }

        let edge_data = masses
            .into_iter()
            .zip(external_shifts)
            .map(|(mass, shift)| {
                let option_mass = if mass == 0.0 { None } else { Some(mass) };
                (option_mass, shift.vector)
            })
            .collect();

        Ok(Self { data: edge_data })
    }
}

#[pymodule]
#[pyo3(name = "momtrop")]
fn momtrop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonEdge>()?;
    m.add_class::<PythonGraph>()?;
    m.add_class::<PythonSampler>()?;
    m.add_class::<PythonSettings>()?;
    m.add_class::<PythonVector>()?;
    m.add_class::<PythonTropicalSampleResult>()?;
    m.add_class::<PythonEdgeData>()?;
    Ok(())
}
