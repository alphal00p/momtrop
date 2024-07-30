use itertools::{izip, Itertools};
use statrs::function::gamma::gamma;

use super::TropicalSubgraphTable;
use crate::matrix::SquareMatrix;
use crate::mimic_rng::MimicRng;
use crate::vector::Vector;
use crate::TropicalSampleResult;
use crate::{float::FloatLike, gamma::inverse_gamma_lr};

fn box_muller<T: FloatLike>(x1: T, x2: T) -> (T, T) {
    let r = (-Into::<T>::into(2.) * x1.ln()).sqrt();
    let theta = Into::<T>::into(2.) * T::PI() * x2;
    (r * theta.cos(), r * theta.sin())
}

pub fn sample<T: FloatLike + Into<f64>>(
    tropical_subgraph_table: &TropicalSubgraphTable,
    x_space_point: &[T],
    loop_signature: &[Vec<isize>],
    edge_data: Vec<(Option<T>, Vector<T>)>,
    print_debug_info: bool,
) -> TropicalSampleResult<T> {
    let num_loops = tropical_subgraph_table.tropical_graph.num_loops;

    let mut mimic_rng = MimicRng::new(x_space_point);
    let permatuhedral_sample =
        permatuhedral_sampling(tropical_subgraph_table, &mut mimic_rng, print_debug_info);

    let l_matrix = compute_l_matrix(&permatuhedral_sample.x, loop_signature);
    let decomposed_l_matrix = l_matrix
        .decompose_for_tropical()
        .unwrap_or_else(|err| panic!("Matrix algorithm failed: {err}"));

    let lambda = Into::<T>::into(inverse_gamma_lr(
        tropical_subgraph_table.tropical_graph.dod,
        Into::<f64>::into(mimic_rng.get_random_number(Some("sample lambda"))),
        50,
        5.0,
    ));

    if print_debug_info {
        println!("lambda: {}", lambda);
    }

    let (edge_masses, edge_shifts): (Vec<T>, Vec<Vector<T>>) = edge_data
        .into_iter()
        .map(|(option_mass, edge_shift)| {
            if let Some(mass) = option_mass {
                (mass, edge_shift)
            } else {
                (T::zero(), edge_shift)
            }
        })
        .unzip();

    let q_vectors = sample_q_vectors(&mut mimic_rng, tropical_subgraph_table.dimension, num_loops);
    let u_vectors = compute_u_vectors(
        &permatuhedral_sample.x,
        loop_signature,
        &edge_shifts,
        tropical_subgraph_table.dimension,
    );

    let v_polynomial = compute_v_polynomial(
        &permatuhedral_sample.x,
        &u_vectors,
        &decomposed_l_matrix.inverse,
        &edge_shifts,
        &edge_masses,
    );

    let loop_momenta = compute_loop_momenta(
        v_polynomial,
        lambda,
        &decomposed_l_matrix.q_transposed_inverse,
        &q_vectors,
        &decomposed_l_matrix.inverse,
        &u_vectors,
        tropical_subgraph_table.dimension,
    );

    let i_trop = Into::<T>::into(tropical_subgraph_table.table.last().unwrap().j_function);

    let gamma_omega = Into::<T>::into(gamma(tropical_subgraph_table.tropical_graph.dod));
    let denom = Into::<T>::into(
        tropical_subgraph_table
            .tropical_graph
            .topology
            .iter()
            .map(|e| gamma(e.weight))
            .product::<f64>(),
    );

    if print_debug_info {
        println!("v: {}", v_polynomial);
        println!("u: {}", decomposed_l_matrix.determinant);
    }

    let u_trop = permatuhedral_sample.u_trop;
    let v_trop = permatuhedral_sample.v_trop;
    let u = decomposed_l_matrix.determinant;
    let v = v_polynomial;
    let prefactor = i_trop * gamma_omega / denom;

    let jacobian = (u_trop / u).powf(Into::<T>::into(
        tropical_subgraph_table.dimension as f64 / 2.0,
    )) * (v_trop / v)
        .powf(Into::<T>::into(tropical_subgraph_table.tropical_graph.dod))
        * prefactor;

    TropicalSampleResult {
        loop_momenta,
        u_trop,
        v_trop,
        u,
        v,
        prefactor,
        jacobian,
    }
}

struct PermatuhedralSamplingResult<T> {
    x: Vec<T>,
    u_trop: T,
    v_trop: T,
}

/// This function returns the feynman parameters for a given graph and sample point, it also computes u_trop and v_trop.
/// A rescaling is performed for numerical stability, with this rescaling u_trop and v_trop always evaluate to 1.
fn permatuhedral_sampling<T: FloatLike>(
    tropical_subgraph_table: &TropicalSubgraphTable,
    rng: &mut MimicRng<T>,
    print_debug_info: bool,
) -> PermatuhedralSamplingResult<T> {
    let mut kappa = T::one();
    let mut x_vec = vec![T::zero(); tropical_subgraph_table.tropical_graph.topology.len()];
    let mut u_trop = T::one();
    let mut v_trop = T::one();

    let mut graph = tropical_subgraph_table
        .tropical_graph
        .get_full_subgraph_id();

    while !graph.is_empty() {
        // this saves a random variable
        let (edge, graph_without_edge) = if graph.has_one_edge() {
            let edge = graph.contains_edges()[0];
            let graph_without_edge = graph.pop_edge(edge);
            (edge, graph_without_edge)
        } else {
            tropical_subgraph_table.sample_edge(rng.get_random_number(Some("sample_edge")), &graph)
        };

        x_vec[edge] = kappa;

        if tropical_subgraph_table.table[graph.get_id()].mass_momentum_spanning
            && !tropical_subgraph_table.table[graph_without_edge.get_id()].mass_momentum_spanning
        {
            v_trop = x_vec[edge];
        }

        if tropical_subgraph_table.table[graph_without_edge.get_id()].loop_number
            < tropical_subgraph_table.table[graph.get_id()].loop_number
        {
            u_trop *= x_vec[edge];
        }

        // Terminate early, so we do not waste a random variable in the final step
        graph = graph_without_edge;
        if graph.is_empty() {
            break;
        }

        let xi = rng.get_random_number(Some("sample xi"));
        kappa *= xi.powf(
            Into::<T>::into(tropical_subgraph_table.table[graph.get_id()].generalized_dod).inv(),
        );
    }

    let xi_trop = u_trop * v_trop;

    // perform rescaling for numerical stability
    let target = u_trop.powf(Into::<T>::into(
        -(tropical_subgraph_table.dimension as f64 / 2.0),
    )) * (u_trop / xi_trop)
        .powf(Into::<T>::into(tropical_subgraph_table.tropical_graph.dod));

    let loop_number = tropical_subgraph_table.table.last().unwrap().loop_number;
    let scaling = target.powf(
        Into::<T>::into(
            tropical_subgraph_table.dimension as f64 / 2.0 * loop_number as f64
                + tropical_subgraph_table.tropical_graph.dod,
        )
        .inv(),
    );

    if print_debug_info {
        println!("feynman parameters before rescaling: {:?}", x_vec);
    }

    x_vec.iter_mut().for_each(|x| *x *= scaling);

    if print_debug_info {
        println!("sampled feynman parameters: {:?}", x_vec);
        println!("u_trop before rescaling: {}", u_trop);
        println!("v_trop before rescaling: {}", v_trop);
    }

    u_trop = T::one();
    v_trop = T::one();

    PermatuhedralSamplingResult {
        x: x_vec,
        u_trop,
        v_trop,
    }
}

/// Compute the L x L matrix from the feynman parameters and the signature matrix
#[inline]
fn compute_l_matrix<T: FloatLike>(x_vec: &[T], signature_matrix: &[Vec<isize>]) -> SquareMatrix<T> {
    let num_edges = signature_matrix.len();
    let num_loops = signature_matrix[0].len();

    let mut temp_l_matrix = SquareMatrix::new_zeros(num_loops);

    for i in 0..num_loops {
        for j in 0..num_loops {
            for e in 0..num_edges {
                temp_l_matrix[(i, j)] += x_vec[e]
                    * Into::<T>::into((signature_matrix[e][i] * signature_matrix[e][j]) as f64);
            }
        }
    }

    temp_l_matrix
}

/// Sample Gaussian distributed vectors, using the Box-Muller transform
#[inline]
fn sample_q_vectors<T: FloatLike>(
    rng: &mut MimicRng<T>,
    dimension: usize,
    num_loops: usize,
) -> Vec<Vector<T>> {
    let token = Some("box muller");
    let num_variables = dimension * num_loops;

    let num_uniform_variables = num_variables + num_variables % 2;
    let gaussians = (0..num_uniform_variables / 2).flat_map(|_| {
        let (box_muller_1, box_muller_2) =
            box_muller(rng.get_random_number(token), rng.get_random_number(token));

        [box_muller_1, box_muller_2]
    });

    #[allow(clippy::useless_conversion)] // without the conversion I get an error
    (0..num_loops)
        .zip(gaussians.chunks(dimension).into_iter())
        .map(|(_, chunk)| Vector::from_vec(chunk.collect_vec()))
        .collect_vec()
}

/// Compute the vectors u, according to the formula in the notes
#[inline]
fn compute_u_vectors<T: FloatLike>(
    x_vec: &[T],
    signature_marix: &[Vec<isize>],
    edge_shifts: &[Vector<T>],
    dimension: usize,
) -> Vec<Vector<T>> {
    let num_loops = signature_marix[0].len();
    let num_edges = signature_marix.len();

    (0..num_loops)
        .map(|l| {
            (0..num_edges).fold(Vector::new(dimension), |acc: Vector<T>, e| {
                &acc + &(&edge_shifts[e]
                    * (x_vec[e] * Into::<T>::into(signature_marix[e][l] as f64)))
            })
        })
        .collect_vec()
}

/// Compute the polynomial v, according to the formula in the notes
#[inline]
fn compute_v_polynomial<T: FloatLike>(
    x_vec: &[T],
    u_vectors: &[Vector<T>],
    inverse_l: &SquareMatrix<T>,
    edge_shifts: &[Vector<T>],
    edge_masses: &[T],
) -> T {
    let num_loops = inverse_l.get_dim();

    let term_1 = izip!(x_vec, edge_masses, edge_shifts)
        .map(|(&x_e, &mass, shift)| x_e * (mass * mass + shift.squared()))
        .sum::<T>();

    let term_2 = (0..num_loops)
        .cartesian_product(0..num_loops)
        .map(|(i, j)| u_vectors[i].dot(&u_vectors[j]) * inverse_l[(i, j)])
        .sum::<T>();

    term_1 - term_2
}

/// Compute the loop momenta, according to the formula in the notes
#[inline]
fn compute_loop_momenta<T: FloatLike>(
    v: T,
    lambda: T,
    q_t_inverse: &SquareMatrix<T>,
    q_vectors: &[Vector<T>],
    l_inverse: &SquareMatrix<T>,
    u_vectors: &[Vector<T>],
    dimension: usize,
) -> Vec<Vector<T>> {
    let num_loops = q_t_inverse.get_dim();
    let prefactor = (v / lambda * Into::<T>::into(0.5)).sqrt();

    (0..num_loops)
        .map(|l| {
            q_vectors.iter().zip(u_vectors.iter()).enumerate().fold(
                Vector::new(dimension),
                |acc, (l_prime, (q, u))| {
                    let q_part: Vector<T> = q * (prefactor * q_t_inverse[(l, l_prime)]);
                    let u_part: Vector<T> = u * l_inverse[(l, l_prime)];

                    &(&acc + &q_part) - &u_part
                },
            )
        })
        .collect_vec()
}
