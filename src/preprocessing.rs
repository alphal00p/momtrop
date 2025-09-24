use core::f64;
use std::vec;

use ahash::HashSet;
use bincode::{Decode, Encode};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::gamma;

use crate::{Graph, MAX_EDGES, float::MomTropFloat};
use clarabel::{
    algebra::CscMatrix,
    solver::{
        DefaultSettings, DefaultSolver, IPSolver,
        SupportedConeT::{NonnegativeConeT, ZeroConeT},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct TropicalGraph {
    pub dod: f64,
    pub topology: Vec<TropicalEdge>,
    pub num_massive_edges: usize,
    pub external_vertices: Vec<u8>,
    pub num_loops: usize,
}

impl TropicalGraph {
    pub fn from_graph(graph: Graph, dimension: usize) -> Self {
        assert!(
            graph.edges.len() <= MAX_EDGES,
            "Graph has more than 64 edges"
        );

        let topology = graph
            .edges
            .into_iter()
            .enumerate()
            .map(|(id, edge)| TropicalEdge {
                edge_id: id as u8,
                left: edge.vertices.0,
                right: edge.vertices.1,
                weight: edge.weight,
                is_massive: edge.is_massive,
            })
            .collect_vec();

        let all_edges = (0..topology.len()).collect_vec();

        let mut res = Self {
            dod: 0.0,
            topology,
            num_massive_edges: 0,
            external_vertices: graph.externals,
            num_loops: 0,
        };

        let weight_sum = res.compute_weight_sum(&all_edges);
        let loop_number = res.get_loop_number(&all_edges);
        let num_massive_edges = res.topology.iter().filter(|edge| edge.is_massive).count();

        let dod = weight_sum - (loop_number as f64 * dimension as f64) / 2.;

        res.dod = dod;
        res.num_massive_edges = num_massive_edges;
        res.num_loops = loop_number;

        res
    }

    #[inline]
    pub fn get_full_subgraph_id(&self) -> TropicalSubGraphId {
        TropicalSubGraphId::new(self.topology.len())
    }

    /// subgraph gamma of a parent graph G is mass-spanning if it contains all massive propagators of G.
    /// and momentum-spanning if it has a connected component that contains all external vertices of G.
    /// A mass-momentum-spanning subgraph is both mass-spanning and momentum-spanning.
    pub fn is_mass_momentum_spanning(&self, edges_in_subgraph: &[usize]) -> bool {
        let num_massive_edges = edges_in_subgraph
            .iter()
            .filter(|&&i| self.topology[i].is_massive)
            .count();

        let is_mass_spanning = num_massive_edges == self.num_massive_edges;

        let connected_compoenents = self.get_connected_components(edges_in_subgraph);

        let is_momentum_spanning = connected_compoenents.iter().any(|component| {
            self.external_vertices.iter().all(|&v| {
                component
                    .contains_edges()
                    .any(|i| self.topology[i].contains_vertex(v))
            })
        });

        is_mass_spanning && is_momentum_spanning
    }

    /// Get all connected components of a graph, used to compute loop number of possible disconnected graph
    pub fn get_connected_components(&self, edges_in_subgraph: &[usize]) -> Vec<TropicalSubGraphId> {
        let num_edges_in_subgraph = edges_in_subgraph.len();

        if num_edges_in_subgraph == 0 {
            return vec![];
        }

        let mut visited_edges: HashSet<usize> = HashSet::default();
        let mut connected_components: Vec<HashSet<usize>> = vec![];
        let mut current_component: HashSet<usize> = HashSet::default();

        // start search in the first edge
        let mut current_edges = vec![edges_in_subgraph[0]];

        visited_edges.insert(current_edges[0]);
        current_component.insert(current_edges[0]);

        while num_edges_in_subgraph
            > connected_components
                .iter()
                .map(std::collections::HashSet::len)
                .sum::<usize>()
        {
            let neighbours = current_edges
                .iter()
                .flat_map(|&edge_id| {
                    self.get_neighbouring_edges_in_subgraph(edge_id, edges_in_subgraph)
                })
                .collect::<Vec<usize>>();

            let mut current_component_grown = false;
            for &neighbour in &neighbours {
                current_edges.push(neighbour);
                visited_edges.insert(neighbour);
                if current_component.insert(neighbour) {
                    current_component_grown = true;
                };
            }

            if !current_component_grown {
                connected_components.push(current_component);
                current_component = HashSet::default();
                current_edges.clear();
                for &edge_id in edges_in_subgraph {
                    if !visited_edges.contains(&edge_id) {
                        current_edges.push(edge_id);
                        visited_edges.insert(edge_id);
                        current_component.insert(edge_id);
                        break;
                    }
                }
            }
        }

        connected_components
            .into_iter()
            .map(|component| {
                TropicalSubGraphId::from_edge_list(
                    &component.into_iter().collect::<Vec<usize>>(),
                    self.topology.len(),
                )
            })
            .collect()
    }

    /// get the loop number of a potentially disconnected grraph
    fn get_loop_number(&self, edges_in_subgraph: &[usize]) -> usize {
        if edges_in_subgraph.is_empty() {
            return 0;
        }

        let connected_components = self.get_connected_components(edges_in_subgraph);

        connected_components
            .iter()
            .map(|c| self.get_loop_number_of_connected_component(c))
            .sum()
    }

    /// Get the loop number of a connected graph, by Euler's formula
    fn get_loop_number_of_connected_component(&self, subgraph_id: &TropicalSubGraphId) -> usize {
        let edges_in_connected_subgraph = subgraph_id.contains_edges();
        let mut vertices: HashSet<u8> = HashSet::default();

        let mut num_edges = 0;
        for edge in edges_in_connected_subgraph {
            vertices.insert(self.topology[edge].left);
            vertices.insert(self.topology[edge].right);
            num_edges += 1;
        }

        let num_vertices = vertices.len();
        1 + num_edges - num_vertices
    }

    /// Sum all the weights of the edges in the subgraph
    fn compute_weight_sum(&self, edges_in_subgraph: &[usize]) -> f64 {
        edges_in_subgraph
            .iter()
            .map(|&i| self.topology[i].weight)
            .sum()
    }

    /// Get all the edges that share a vertex with a given edge
    fn get_neighbouring_edges_in_subgraph(
        &self,
        edge_id: usize,
        edges_in_subgraph: &[usize],
    ) -> Vec<usize> {
        edges_in_subgraph
            .iter()
            .filter(|&&i| self.are_neighbours(edge_id, i))
            .copied()
            .collect()
    }

    /// Check if two edges share a vertex.
    fn are_neighbours(&self, edge_id_1: usize, edge_id_2: usize) -> bool {
        self.topology[edge_id_1].contains_vertex(self.topology[edge_id_2].left)
            || self.topology[edge_id_1].contains_vertex(self.topology[edge_id_2].right)
    }

    /// Definition of the j-function for a subgraph, described in the tropical sampling papers.
    fn recursive_fill_j_function(
        subgraph_id: &TropicalSubGraphId,
        table: &mut Vec<OptionTropicalSubgraphTableEntry>,
    ) -> f64 {
        if subgraph_id.is_empty() {
            let j_function = 1.0;
            table[subgraph_id.id].j_function = Some(j_function);
            j_function
        } else {
            // this guards against infinite recursion, for some reason it occurs in the 4-loop ladder
            if let Some(j_function) = table[subgraph_id.id].j_function {
                return j_function;
            }

            let edges_in_subgraph = subgraph_id.contains_edges();
            let subgraphs = edges_in_subgraph.map(|e| subgraph_id.pop_edge(e));

            let j_function = subgraphs
                .map(|g| {
                    TropicalGraph::recursive_fill_j_function(&g, table)
                        / table[g.id].generalized_dod.unwrap()
                })
                .sum();
            table[subgraph_id.id].j_function = Some(j_function);
            j_function
        }
    }

    fn generate_stage_1(&self) -> Vec<OptionTropicalSubgraphTableEntry> {
        let num_edges = self.topology.len();
        let powerset_size = 2usize.pow(num_edges as u32);

        // allocate the subgraph table
        let mut option_subgraph_table =
            vec![OptionTropicalSubgraphTableEntry::all_none(); powerset_size];

        // create iterator over all subgraphs
        let subgraph_iterator =
            (0..powerset_size).map(|i| TropicalSubGraphId::from_id(i, num_edges));

        for subgraph in subgraph_iterator {
            let edges_in_subgraph = subgraph.contains_edges().collect_vec();

            // check the mass-momentum spanning property
            let is_mass_momentum_spanning = self.is_mass_momentum_spanning(&edges_in_subgraph);

            option_subgraph_table[subgraph.id].mass_momentum_spanning =
                Some(is_mass_momentum_spanning);

            let loop_number = self.get_loop_number(&edges_in_subgraph);

            option_subgraph_table[subgraph.id].loop_number = Some(loop_number as u8);
        }

        option_subgraph_table
    }

    fn optimize_edge_weights(
        &mut self,
        dimension: usize,
        progress: &[OptionTropicalSubgraphTableEntry],
        target_omega: f64,
    ) {
        let num_edges = self.topology.len();
        let input_size = num_edges + 1;

        let full_id = self.get_full_subgraph_id();

        let mut q = vec![0.0; num_edges];
        q.push(-1.0);

        let mut p = Vec::with_capacity(input_size);

        for i in 0..num_edges {
            let mut row = vec![0.0; input_size];
            row[i] = 1.0;
            p.push(row);
        }

        p.push(vec![0.0; input_size]);

        let p = CscMatrix::from(&p);

        let output_size = 2u64.pow(num_edges as u32) as usize; // number of constraints
        let mut a = Vec::with_capacity(output_size);
        let mut b = Vec::with_capacity(output_size);

        for (subgraph_usize, entry) in progress.iter().enumerate() {
            let subgraph = TropicalSubGraphId::from_id(subgraph_usize, num_edges);
            let is_mass_momentum_spanning = entry
                .mass_momentum_spanning
                .unwrap_or_else(|| unreachable!());

            let loop_number = entry.loop_number.unwrap_or_else(|| unreachable!());

            if subgraph.is_empty() {
                continue;
            } else if subgraph == full_id {
                let mut row = vec![-1.0; input_size - 1];
                row.push(0.0);
                a.push(row);
                b.push(self.num_loops as f64 * dimension as f64 / -2.0);
            } else {
                let mut row = vec![0.0; input_size];
                for edge in subgraph.contains_edges() {
                    row[edge] = -1.0;
                }

                if is_mass_momentum_spanning {
                    for edge in 0..num_edges {
                        row[edge] += 1.0;
                    }
                }

                a.push(row);

                if is_mass_momentum_spanning {
                    b.push((self.num_loops as u8 - loop_number) as f64 * dimension as f64 / 2.0);
                } else {
                    b.push(-(loop_number as f64 * dimension as f64) / 2.0);
                }
            }
        }

        let mut last_row = vec![0.0; input_size - 1];
        last_row.push(1.0);
        a.push(last_row);
        b.push(target_omega);

        //display_matrix(&a);
        //display_column_vector(&b);

        let a = CscMatrix::from(&a);
        let cones = [NonnegativeConeT(output_size - 1), ZeroConeT(1)];
        let settings = DefaultSettings::default();

        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
        solver.solve();

        let solution = solver.solution.x;
        let new_edge_weights = &solution[0..num_edges];

        let mut weight_sum = 0.0;
        for (i, edge) in self.topology.iter_mut().enumerate() {
            edge.weight = new_edge_weights[i];
            weight_sum += edge.weight;
            println!("Edge {} new weight: {}", i, edge.weight);
        }

        self.dod = weight_sum - (self.num_loops as f64 * dimension as f64) / 2.0;
    }

    fn generate_stage_2(
        self,
        dimension: usize,
        mut progress: Vec<OptionTropicalSubgraphTableEntry>,
    ) -> Result<TropicalSubgraphTable, String> {
        if self.dod < 0.0 {
            return Err(format!("Graph has negative DoD: {}", self.dod));
        }

        let full_subgraph_id = self.get_full_subgraph_id();

        for (subgraph_usize, table_entry) in progress.iter_mut().enumerate() {
            let subgraph = TropicalSubGraphId::from_id(subgraph_usize, self.topology.len());
            let edges_in_subgraph = subgraph.contains_edges().collect_vec();

            let weight_sum = self.compute_weight_sum(&edges_in_subgraph);
            let is_mass_momentum_spanning =
                table_entry.mass_momentum_spanning.unwrap_or_else(|| {
                    unreachable!("mass-momentum spanning not set for subgraph {subgraph:?}")
                });

            let loop_number = table_entry
                .loop_number
                .unwrap_or_else(|| unreachable!("loop number not set for subgraph {subgraph:?}"));

            let generalized_dod = if !subgraph.is_empty() {
                if is_mass_momentum_spanning {
                    weight_sum - loop_number as f64 * dimension as f64 / 2.0 - self.dod
                } else {
                    weight_sum - loop_number as f64 * dimension as f64 / 2.0
                }
            } else {
                1.0
            };

            if generalized_dod <= 0.0 && !subgraph.is_empty() && subgraph != full_subgraph_id {
                return Err(format!(
                    "Generalized DoD: {generalized_dod} is negative for subgraph {subgraph:?}\n
                    loop number: {loop_number}, mass-momentum spanning: {is_mass_momentum_spanning}, weight sum: {weight_sum}"
                ));
            }

            table_entry.generalized_dod = Some(generalized_dod);
        }

        TropicalGraph::recursive_fill_j_function(&full_subgraph_id, &mut progress);

        let gamma_omega = gamma(self.dod);
        let denom = self
            .topology
            .iter()
            .map(|e| gamma(e.weight))
            .product::<f64>();

        let gamma_ratio = gamma_omega / denom;
        let pi_factor = f64::consts::PI.powf((dimension * self.num_loops) as f64 / 2.);

        let table = progress
            .into_iter()
            .map(OptionTropicalSubgraphTableEntry::to_entry)
            .collect_vec();

        let i_tr = table.last().unwrap().j_function;

        let cached_factor = i_tr * gamma_ratio * pi_factor;

        Ok(TropicalSubgraphTable {
            table,
            dimension,
            cached_factor,
            tropical_graph: self,
        })
    }
}

#[allow(unused)]
fn display_matrix(matrix: &Vec<Vec<f64>>) {
    for row in matrix {
        println!("{:?}", row);
    }
}

#[allow(unused)]
fn display_column_vector(v: &Vec<f64>) {
    for entry in v {
        println!("{:?}", entry);
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Encode, Decode)]
pub struct TropicalEdge {
    edge_id: u8,
    left: u8,
    right: u8,
    pub weight: f64,
    is_massive: bool,
}

impl TropicalEdge {
    fn contains_vertex(&self, vertex: u8) -> bool {
        self.left == vertex || self.right == vertex
    }
}

/// The bits in the id represent whether the corresponding edge is in the subgraph
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TropicalSubGraphId {
    id: usize,
    num_edges: usize,
}

impl TropicalSubGraphId {
    #[inline]
    pub fn get_id(&self) -> usize {
        self.id
    }

    fn new(num_edges: usize) -> Self {
        Self {
            id: (1 << num_edges) - 1,
            num_edges,
        }
    }

    fn from_id(id: usize, num_edges: usize) -> Self {
        Self { id, num_edges }
    }

    /// remove an edge from the subgraph
    pub fn pop_edge(&self, edge_id: usize) -> Self {
        Self {
            id: self.id ^ (1 << edge_id),
            num_edges: self.num_edges,
        }
    }

    /// The 0 id as all bits set to 0, so it contains no edges and thus represents the empty graph
    pub fn is_empty(&self) -> bool {
        self.id == 0
    }

    /// Check wheter the subgraph contains a specific edge
    fn has_edge(&self, edge_id: usize) -> bool {
        self.id & (1 << edge_id) != 0
    }

    /// Get the edges contained in the subgraph
    pub fn contains_edges(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.num_edges).filter(|&i| self.has_edge(i))
    }

    /// Check if the subgraph contains only one edge, this is a special case in the tropical sampling algorithm
    pub fn has_one_edge(&self) -> bool {
        self.id.count_ones() == 1
    }

    /// Create a subgraph id from a list of edges
    fn from_edge_list(edge_list: &[usize], num_edges: usize) -> Self {
        let mut id = 0;
        for &edge_id in edge_list {
            id |= 1 << edge_id;
        }
        Self { id, num_edges }
    }
}

/// Helper struct for generation, to store entries before computing the value of the j_function (Which requires the value of the other quantities to be present)
#[derive(Debug, Clone, Copy)]
struct OptionTropicalSubgraphTableEntry {
    loop_number: Option<u8>,
    mass_momentum_spanning: Option<bool>,
    j_function: Option<f64>,
    generalized_dod: Option<f64>,
}

impl OptionTropicalSubgraphTableEntry {
    fn all_none() -> Self {
        Self {
            loop_number: None,
            mass_momentum_spanning: None,
            j_function: None,
            generalized_dod: None,
        }
    }

    /// Check if all fields are set, and panic if it has failed
    fn to_entry(self) -> TropicalSubgraphTableEntry {
        TropicalSubgraphTableEntry {
            loop_number: self
                .loop_number
                .unwrap_or_else(|| panic!("loop number not set")),
            mass_momentum_spanning: self
                .mass_momentum_spanning
                .unwrap_or_else(|| panic!("mass-momentum spanning not set")),
            j_function: self
                .j_function
                .unwrap_or_else(|| panic!("j-function not set")),
            generalized_dod: self
                .generalized_dod
                .unwrap_or_else(|| panic!("generalized dod not set")),
        }
    }
}

/// Data that needs to be stored for each subgraph
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub struct TropicalSubgraphTableEntry {
    pub loop_number: u8,
    pub mass_momentum_spanning: bool,
    pub j_function: f64,
    pub generalized_dod: f64,
}

/// The list of data for all subgraphs, indexed using the TropicalSubGraphId
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct TropicalSubgraphTable {
    pub table: Vec<TropicalSubgraphTableEntry>,
    pub dimension: usize,
    pub tropical_graph: TropicalGraph,
    pub cached_factor: f64,
}

impl TropicalSubgraphTable {
    pub(crate) fn generate_from_tropical(
        mut tropical_graph: TropicalGraph,
        dimension: usize,
        target_omega: Option<f64>,
    ) -> Result<Self, String> {
        let progress = tropical_graph.generate_stage_1();
        if let Some(target_omega) = target_omega {
            tropical_graph.optimize_edge_weights(dimension, &progress, target_omega);
        }
        tropical_graph.generate_stage_2(dimension, progress)
    }

    /// sample an edge from a subgraph, according to the relevant probability distribution, returns the edge and the subgraph without the edge for later use
    /// Panics if the uniform random number is greater than one or the probability distribution is incorrectly normalized.
    pub fn sample_edge<T: MomTropFloat>(
        &self,
        uniform: &T,
        subgraph: &TropicalSubGraphId,
    ) -> (usize, TropicalSubGraphId) {
        let edges_in_subgraph = subgraph.contains_edges();
        let j = uniform.from_f64(self.table[subgraph.id].j_function);

        let mut cum_sum = uniform.zero();
        for edge in edges_in_subgraph {
            let graph_without_edge = subgraph.pop_edge(edge);
            let p_e = uniform.from_f64(self.table[graph_without_edge.id].j_function)
                / &j
                / uniform.from_f64(self.table[graph_without_edge.id].generalized_dod);
            cum_sum += &p_e;
            if &cum_sum >= uniform {
                return (edge, graph_without_edge);
            }
        }

        panic!(
            "Sampling could not sample edge, with uniform_random_number: {:?}, cumulative sum evaluated to: {:?}",
            uniform, cum_sum
        );
    }

    pub fn get_num_variables(&self) -> usize {
        let num_edges = self.tropical_graph.topology.len();

        let loop_number = self
            .tropical_graph
            .get_loop_number(&(0..num_edges).collect_vec());

        let num_gaussian_variables = loop_number * self.dimension;

        2 * num_edges - 1 + num_gaussian_variables + num_gaussian_variables % 2
    }

    pub fn get_smallest_dod(&self) -> f64 {
        self.table
            .iter()
            .map(|entry| entry.generalized_dod)
            .filter(|&x| x > 0.0)
            .reduce(f64::min)
            .unwrap()
    }
}

// some tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, assert_approx_eq};

    const TOLERANCE: f64 = 1e-14;
    // panics with useful error message

    mod subgraph_id {
        use itertools::Itertools;

        use crate::preprocessing::TropicalSubGraphId;
        #[test]
        fn test_get_id() {
            let id = TropicalSubGraphId::new(4);
            assert_eq!(id.get_id(), 15);
        }

        #[test]
        fn test_new() {
            let id = TropicalSubGraphId::new(2);
            assert_eq!(id.num_edges, 2);
            assert_eq!(id.get_id(), 3);
        }

        #[test]
        fn test_contains_edges() {
            let id = TropicalSubGraphId::new(2);
            let edges = id.contains_edges().collect_vec();
            assert_eq!(edges.len(), 2);
            assert!(edges.contains(&0));
            assert!(edges.contains(&1));
        }

        #[test]
        fn test_pop_edge() {
            let mut id = TropicalSubGraphId::new(3);
            id = id.pop_edge(2);

            let edges = id.contains_edges().collect_vec();

            assert_eq!(edges.len(), 2);
            assert!(edges.contains(&0));
            assert!(edges.contains(&1));
        }

        #[test]
        fn test_has_one_edge() {
            let mut id = TropicalSubGraphId::new(3);
            id = id.pop_edge(2);
            id = id.pop_edge(1);

            assert!(id.has_one_edge())
        }

        #[test]
        fn test_is_empty() {
            let mut id = TropicalSubGraphId::new(3);
            id = id.pop_edge(2);
            id = id.pop_edge(1);
            id = id.pop_edge(0);

            assert!(id.is_empty())
        }

        #[test]
        fn test_has_edge() {
            let subgraph_id = TropicalSubGraphId::new(3);

            assert!(subgraph_id.has_edge(0));
            assert!(subgraph_id.has_edge(1));
            assert!(subgraph_id.has_edge(2));

            let subgraph_id = subgraph_id.pop_edge(1);

            assert!(subgraph_id.has_edge(0));
            assert!(!subgraph_id.has_edge(1));
            assert!(subgraph_id.has_edge(2));
        }

        #[test]
        fn test_from_edge_list() {
            let subgraph_id = TropicalSubGraphId::from_edge_list(&[0, 1, 3], 4);
            let edges = subgraph_id.contains_edges().collect_vec();
            assert_eq!(edges.len(), 3);

            assert!(subgraph_id.has_edge(0));
            assert!(subgraph_id.has_edge(1));
            assert!(subgraph_id.has_edge(3));
            assert!(!subgraph_id.has_edge(2));
        }
    }

    fn sunrise_graph() -> Graph {
        Graph {
            edges: vec![
                Edge {
                    vertices: (0, 1),
                    is_massive: false,
                    weight: 1.0,
                },
                Edge {
                    vertices: (0, 1),
                    is_massive: false,
                    weight: 1.0,
                },
                Edge {
                    vertices: (0, 1),
                    is_massive: false,
                    weight: 1.0,
                },
            ],
            externals: vec![0, 1],
        }
    }

    fn massive_sunrise_graph() -> Graph {
        Graph {
            edges: vec![
                Edge {
                    vertices: (0, 1),
                    is_massive: true,
                    weight: 1.0,
                },
                Edge {
                    vertices: (0, 1),
                    is_massive: true,
                    weight: 1.0,
                },
                Edge {
                    vertices: (0, 1),
                    is_massive: true,
                    weight: 1.0,
                },
            ],
            externals: vec![0, 1],
        }
    }

    fn double_triangle_graph() -> Graph {
        Graph {
            edges: vec![
                Edge {
                    vertices: (0, 2),
                    is_massive: false,
                    weight: 1.0,
                },
                Edge {
                    vertices: (0, 3),
                    is_massive: false,
                    weight: 1.0,
                },
                Edge {
                    vertices: (2, 3),
                    is_massive: false,
                    weight: 1.0,
                },
                Edge {
                    vertices: (2, 1),
                    is_massive: false,
                    weight: 1.0,
                },
                Edge {
                    vertices: (3, 1),
                    is_massive: false,
                    weight: 1.0,
                },
            ],
            externals: vec![0, 1],
        }
    }

    #[test]
    fn test_from_graph() {
        let sunrise_graph = sunrise_graph();
        let tropical_sunrise = TropicalGraph::from_graph(sunrise_graph, 3);

        assert_eq!(tropical_sunrise.topology.len(), 3);
        assert_eq!(tropical_sunrise.dod, 0.0);
        assert_eq!(tropical_sunrise.num_loops, 2);
        assert_eq!(tropical_sunrise.num_massive_edges, 0);

        for edge in tropical_sunrise.topology.iter() {
            assert_eq!(edge.left, 0);
            assert_eq!(edge.right, 1);
            assert!(!edge.is_massive);
            assert_eq!(edge.weight, 1.0);
        }
    }

    #[test]
    fn test_get_full_subgraph_id() {
        let sunrise_graph = sunrise_graph();
        let tropical_sunrise = TropicalGraph::from_graph(sunrise_graph, 3);
        let full_subgraph_id = tropical_sunrise.get_full_subgraph_id();
        assert_eq!(full_subgraph_id.get_id(), 7);
    }

    #[test]
    fn test_is_mass_momentum_spanning() {
        let sunrise_graph = sunrise_graph();
        let tropical_sunrise = TropicalGraph::from_graph(sunrise_graph, 3);

        assert!(tropical_sunrise.is_mass_momentum_spanning(&[0, 1, 2]));
        assert!(tropical_sunrise.is_mass_momentum_spanning(&[0, 1]));
        assert!(tropical_sunrise.is_mass_momentum_spanning(&[0, 2]));
        assert!(tropical_sunrise.is_mass_momentum_spanning(&[1, 2]));
        assert!(tropical_sunrise.is_mass_momentum_spanning(&[0]));
        assert!(tropical_sunrise.is_mass_momentum_spanning(&[1]));
        assert!(tropical_sunrise.is_mass_momentum_spanning(&[1]));

        let massive_sunrise_graph = massive_sunrise_graph();
        let massive_sunrise = TropicalGraph::from_graph(massive_sunrise_graph, 3);

        assert!(massive_sunrise.is_mass_momentum_spanning(&[0, 1, 2]));
        assert!(!massive_sunrise.is_mass_momentum_spanning(&[0, 1]));
        assert!(!massive_sunrise.is_mass_momentum_spanning(&[0, 2]));
        assert!(!massive_sunrise.is_mass_momentum_spanning(&[1, 2]));
        assert!(!massive_sunrise.is_mass_momentum_spanning(&[0]));
        assert!(!massive_sunrise.is_mass_momentum_spanning(&[1]));
        assert!(!massive_sunrise.is_mass_momentum_spanning(&[1]));
    }

    #[test]
    fn test_connected_components() {
        let topology = vec![
            TropicalEdge {
                edge_id: 0,
                left: 0,
                right: 1,
                weight: 1.0,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 1,
                left: 1,
                right: 0,
                weight: 1.0,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 2,
                left: 2,
                right: 3,
                weight: 1.0,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 3,
                left: 3,
                right: 2,
                weight: 1.0,
                is_massive: false,
            },
        ];

        let tropical_graph = TropicalGraph {
            dod: 0.0,
            topology,
            num_massive_edges: 0,
            external_vertices: vec![0, 1, 2, 3],
            num_loops: 2,
        };

        let components = tropical_graph.get_connected_components(&[0, 1, 2, 3]);
        assert_eq!(components.len(), 2);

        let components = tropical_graph.get_connected_components(&[0, 1]);
        assert_eq!(components.len(), 1);

        let components = tropical_graph.get_connected_components(&[0, 2]);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_get_loop_number_of_connected_component() {
        let graph = double_triangle_graph();

        let tropical_graph = TropicalGraph::from_graph(graph, 3);

        let subgraph_id = tropical_graph.get_full_subgraph_id();
        let loop_number = tropical_graph.get_loop_number_of_connected_component(&subgraph_id);
        assert_eq!(loop_number, 2);
    }

    #[test]
    fn test_compute_weight_sum() {
        let graph = double_triangle_graph();
        let tropical_graph = TropicalGraph::from_graph(graph, 3);

        let weight_sum = tropical_graph.compute_weight_sum(&[0, 1, 2, 3, 4]);
        assert_eq!(weight_sum, 5.0);

        let weight_sum = tropical_graph.compute_weight_sum(&[0, 3, 4]);
        assert_eq!(weight_sum, 3.0);

        let triangle_with_different_weights = Graph {
            edges: vec![
                Edge {
                    vertices: (0, 1),
                    is_massive: false,
                    weight: 1.0,
                },
                Edge {
                    vertices: (1, 2),
                    is_massive: false,
                    weight: 2.0,
                },
                Edge {
                    vertices: (2, 0),
                    is_massive: false,
                    weight: 3.0,
                },
            ],
            externals: vec![],
        };

        let trop_tri = TropicalGraph::from_graph(triangle_with_different_weights, 3);
        let weight_sum = trop_tri.compute_weight_sum(&[0, 1]);
        assert_eq!(weight_sum, 3.0);

        let weight_sum = trop_tri.compute_weight_sum(&[0, 2]);
        assert_eq!(weight_sum, 4.0);

        let weight_sum = trop_tri.compute_weight_sum(&[1, 2]);
        assert_eq!(weight_sum, 5.0);
    }

    #[test]
    fn test_loop_number() {
        let topology1 = vec![
            TropicalEdge {
                edge_id: 0,
                left: 0,
                right: 1,
                weight: 1.0,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 1,
                left: 1,
                right: 0,
                weight: 1.0,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 2,
                left: 2,
                right: 3,
                weight: 1.0,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 3,
                left: 3,
                right: 2,
                weight: 1.0,
                is_massive: false,
            },
        ];

        let tropical_graph1 = TropicalGraph {
            dod: 0.0,
            topology: topology1,
            num_massive_edges: 0,
            external_vertices: vec![0, 1, 2, 3],
            num_loops: 2,
        };

        let loop_number = tropical_graph1.get_loop_number(&[0, 1, 2, 3]);
        assert_eq!(loop_number, 2);

        let loop_number = tropical_graph1.get_loop_number(&[0, 1]);
        assert_eq!(loop_number, 1);

        let loop_number = tropical_graph1.get_loop_number(&[0, 2]);
        assert_eq!(loop_number, 0);
    }

    #[test]
    fn test_get_neighbours() {
        let double_triangle = double_triangle_graph();
        let trop = TropicalGraph::from_graph(double_triangle, 3);

        let neighbours = trop.get_neighbouring_edges_in_subgraph(0, &[0, 1, 2, 3, 4]);
        assert_eq!(neighbours.len(), 4);

        assert!(neighbours.contains(&1));
        assert!(neighbours.contains(&2));
        assert!(neighbours.contains(&3));
        assert!(!neighbours.contains(&4))
    }

    #[test]
    fn test_are_neighbours() {
        let trop = TropicalGraph::from_graph(double_triangle_graph(), 3);
        assert!(trop.are_neighbours(0, 1));
        assert!(trop.are_neighbours(0, 2));
        assert!(trop.are_neighbours(0, 3));
        assert!(!trop.are_neighbours(0, 4));

        assert!(trop.are_neighbours(1, 2));
        assert!(!trop.are_neighbours(1, 3));
        assert!(trop.are_neighbours(1, 4));

        assert!(trop.are_neighbours(2, 3));
        assert!(trop.are_neighbours(2, 4));

        assert!(trop.are_neighbours(3, 4));
    }

    // tests compared against the output of feyntrop
    #[test]
    fn test_triangle() {
        let triangle_topology = vec![
            TropicalEdge {
                edge_id: 0,
                left: 0,
                right: 1,
                weight: 0.66,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 1,
                left: 1,
                right: 2,
                weight: 0.66,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 2,
                left: 2,
                right: 0,
                weight: 0.66,
                is_massive: false,
            },
        ];

        let triangle_graph = TropicalGraph {
            dod: 3. * 0.66 - 3. / 2.,
            topology: triangle_topology,
            num_massive_edges: 0,
            external_vertices: vec![0, 1, 2],
            num_loops: 1,
        };

        let subgraph_table = TropicalSubgraphTable::generate_from_tropical(triangle_graph, 3, None)
            .expect("Failed to generate subgraph table");

        //panic!("{:?}", subgraph_table.tropical_graph);

        assert_eq!(subgraph_table.table.len(), 8);

        assert_eq!(
            subgraph_table.table[0],
            TropicalSubgraphTableEntry {
                loop_number: 0,
                mass_momentum_spanning: false,
                j_function: 1.0,
                generalized_dod: 1.0,
            }
        );

        assert_eq!(
            subgraph_table.table[1],
            TropicalSubgraphTableEntry {
                loop_number: 0,
                mass_momentum_spanning: false,
                j_function: 1.0,
                generalized_dod: 0.66,
            }
        );

        assert_eq!(subgraph_table.table[2], subgraph_table.table[1]);
        assert_eq!(subgraph_table.table[4], subgraph_table.table[1]);

        for i in [3, 5, 6] {
            let table = subgraph_table.table[i];
            assert!(table.mass_momentum_spanning);
            assert_eq!(table.loop_number, 0);
            assert_approx_eq(&table.generalized_dod, &0.84, &TOLERANCE);
            assert_approx_eq(&table.j_function, &3.030_303_030_303_03, &TOLERANCE);
        }

        let final_table = subgraph_table.table[7];
        assert!(final_table.mass_momentum_spanning);
        assert_eq!(final_table.loop_number, 1);
        assert_approx_eq(&final_table.generalized_dod, &0.0, &TOLERANCE);
        assert_approx_eq(&final_table.j_function, &10.822_510_822_510_82, &TOLERANCE);
    }

    #[test]
    fn mercedes() {
        let weight = 11. / 14.;
        let externals = vec![0, 2];

        let mercedes_topology = vec![
            TropicalEdge {
                edge_id: 0,
                left: 0,
                right: 1,
                weight,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 1,
                left: 1,
                right: 2,
                weight,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 2,
                left: 2,
                right: 3,
                weight,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 3,
                left: 3,
                right: 0,
                weight,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 4,
                left: 1,
                right: 4,
                weight,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 5,
                left: 2,
                right: 4,
                weight,
                is_massive: false,
            },
            TropicalEdge {
                edge_id: 6,
                left: 3,
                right: 4,
                weight,
                is_massive: false,
            },
        ];

        let gr = TropicalGraph {
            dod: 1.0,
            num_massive_edges: 0,
            topology: mercedes_topology,
            external_vertices: externals,
            num_loops: 3,
        };

        let subgraph_table = TropicalSubgraphTable::generate_from_tropical(gr, 3, None)
            .expect("Failed to generate subgraph table");

        //panic!("{:?}", subgraph_table.tropical_graph);

        let i_tr = subgraph_table.table.last().unwrap().j_function;

        assert_approx_eq(&i_tr, &1_818.303_855_640_347_1, &TOLERANCE);
    }
}
