use itertools::Itertools;
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods, pymodule,
    types::{PyModule, PyModuleMethods},
    Bound, PyResult,
};

use crate::{Edge, Graph, SampleGenerator};

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
    fn new(graph: PythonGraph, loop_signature: Vec<Vec<isize>>) -> PyResult<Self> {
        match graph.graph.build_sampler(loop_signature) {
            Ok(sampler) => Ok(PythonSampler { sampler }),
            Err(error_message) => Err(PyValueError::new_err(error_message)),
        }
    }

    pub fn get_dimension(&self) -> usize {
        self.sampler.get_dimension()
    }
}

#[pymodule]
#[pyo3(name = "momtrop")]
fn momtrop(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonEdge>()?;
    m.add_class::<PythonGraph>()?;
    m.add_class::<PythonSampler>()
}
