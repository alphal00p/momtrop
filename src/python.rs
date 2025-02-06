use itertools::Itertools;
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods, pymodule,
    types::{PyModule, PyModuleMethods},
    Bound, PyResult,
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
}

#[pyclass(name = "Settings")]
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
}

#[pyclass(name = "TropicalSampleResult")]
pub struct PythonTropicalSampleResult {
    result: TropicalSampleResult<f64, 3>,
}

#[pymethods]
impl PythonTropicalSampleResult {
    fn get_loop_momenta(&self) -> Vec<PythonVector> {
        self.result
            .loop_momenta
            .iter()
            .map(|&loop_momentum| PythonVector {
                vector: loop_momentum,
            })
            .collect()
    }

    fn get_jacobian(&self) -> f64 {
        self.result.jacobian
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
    Ok(())
}
