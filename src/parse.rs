use std::collections::HashMap;

use dot_parser::{
    ast::{self},
    canonical,
};

use crate::{Edge, Graph, SampleGenerator};
const ALLOWED_NODE_ATTRS: [&str; 1] = ["is_ext"];
const ALLOWED_EDGE_ATTRS: [&str; 3] = ["weight", "is_massive", "loop_sign"];

#[derive(Clone, Debug)]
struct GraphAndLoopSign {
    graph: Graph,
    loop_signature: Vec<Vec<isize>>,
}

impl TryFrom<&str> for GraphAndLoopSign {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let ast_graph = ast::Graph::try_from(value).map_err(|e| e.to_string())?;
        let dot_parsed_graph = canonical::Graph::from(ast_graph);

        if dot_parsed_graph.nodes.set.len() > 256 {
            return Err("Graphs with more than 256 nodes are not supported".to_string());
        }

        if dot_parsed_graph.edges.set.len() > 64 {
            return Err("Graphs with more than 64 edges are not supported".to_string());
        }

        let mut externals = Vec::<u8>::new();

        let node_list = dot_parsed_graph
            .nodes
            .set
            .iter()
            .enumerate()
            .map(|(node_id, (node_name, node_data))| {
                let node_attr = node_data
                    .attr
                    .elems
                    .iter()
                    .map(|(k, v)| (k.clone().into(), v.clone().into()))
                    .collect::<HashMap<String, String>>();

                if node_attr.len() != node_data.attr.elems.len() {
                    return Err("Duplicate node attributes are not allowed".to_string());
                }

                for key in node_attr.keys() {
                    if !ALLOWED_NODE_ATTRS.contains(&key.as_str()) {
                        return Err(format!("Node attribute '{}' is not known", key));
                    }
                }

                if let Some(is_ext_str) = node_attr.get("is_ext") {
                    let is_ext: bool = is_ext_str
                        .parse()
                        .map_err(|e| format!("failed to parse is_ext to bool: {}", e))?;

                    if is_ext {
                        externals.push(node_id as u8);
                    }
                }

                println!("adding node {} with id {}", node_name, node_id);
                Ok((node_name.clone(), node_id as u8))
            })
            .collect::<Result<HashMap<String, u8>, String>>()?;

        let (edges, loop_signature): (Vec<Edge>, Vec<Vec<isize>>) = dot_parsed_graph
            .edges
            .set
            .iter()
            .map(|edge| {
                let from = *node_list
                    .get(&edge.from)
                    .ok_or_else(|| format!("Edge refers to unknown node '{}'", edge.from))?;
                let to = *node_list
                    .get(&edge.to)
                    .ok_or_else(|| format!("Edge refers to unknown node '{}'", edge.to))?;

                let vertices = (from, to);

                let edge_attr = edge
                    .attr
                    .elems
                    .iter()
                    .map(|(k, v)| (k.clone().into(), v.clone().into()))
                    .collect::<HashMap<String, String>>();

                if edge_attr.len() != edge.attr.elems.len() {
                    return Err("Duplicate edge attributes are not allowed".to_string());
                }

                for key in edge_attr.keys() {
                    if !ALLOWED_EDGE_ATTRS.contains(&key.as_str()) {
                        return Err(format!("Edge attribute '{}' is not known", key));
                    }
                }

                let is_massive = if let Some(is_massive_str) = edge_attr.get("is_massive") {
                    is_massive_str
                        .parse()
                        .map_err(|e| format!("failed to parse is_massive to bool: {}", e))?
                } else {
                    false
                };

                let weight: f64 = if let Some(weight_str) = edge_attr.get("weight") {
                    weight_str
                        .parse()
                        .map_err(|e| format!("failed to parse weight to f64: {}", e))?
                } else {
                    return Err("Edge attribute 'weight' is required".to_string());
                };

                let loop_sign: Vec<isize> = if let Some(loop_sign_str) = edge_attr.get("loop_sign")
                {
                    parse_str_to_sign_comp(loop_sign_str)?
                } else {
                    return Err("Edge attribute 'loop_sign' is required".to_string());
                };

                Ok((
                    Edge {
                        vertices,
                        weight,
                        is_massive,
                    },
                    loop_sign,
                ))
            })
            .collect::<Result<Vec<(Edge, Vec<isize>)>, String>>()?
            .into_iter()
            .unzip();

        let graph = Graph { edges, externals };

        Ok(Self {
            graph,
            loop_signature,
        })
    }
}

impl<const D: usize> TryFrom<&str> for SampleGenerator<D> {
    type Error = String;

    /// note that there is no connection between the order of the nodes in the dot file and the indices in the sampler
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let graph_and_loop_sign = GraphAndLoopSign::try_from(value)?;
        graph_and_loop_sign
            .graph
            .build_sampler(graph_and_loop_sign.loop_signature)
    }
}

fn parse_str_to_sign_comp(input: &str) -> Result<Vec<isize>, String> {
    let trimed = input.split_whitespace().collect::<String>();
    if !trimed.starts_with('[') || !trimed.ends_with(']') {
        return Err("Loop signature must start with '[' and end with ']'".to_string());
    }
    let inner = &trimed[1..trimed.len() - 1];
    let components = inner.split(',').collect::<Vec<_>>();
    components
        .into_iter()
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|e| format!("failed to parse '{}': {}", s, e))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::parse::GraphAndLoopSign;

    #[test]
    fn test_parse() {
        let triangle_dot = "digraph G {
            v1 [is_ext=true];
            v2 [is_ext=true];
            v3 [is_ext=true];
            v1 -> v2 [weight=\"1.0\", is_massive=false, loop_sign=\"[1]\"];
            v2 -> v3 [is_massive=false, weight=\"2.0\", loop_sign=\"[1]\"];
            v3 -> v1 [is_massive=false, weight=\"3.0\", loop_sign=\"[1]\"];
        }";

        let graph = GraphAndLoopSign::try_from(triangle_dot).unwrap();
        println!("{:?}", graph);
    }
}
