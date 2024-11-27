use itertools::{izip, Itertools};

use super::TropicalSubgraphTable;
#[cfg(feature = "log")]
use crate::log::Logger;
use crate::matrix::{MatrixError, SquareMatrix};
use crate::mimic_rng::MimicRng;
use crate::vector::Vector;
use crate::{float::MomTropFloat, gamma::inverse_gamma_lr};
use crate::{Metadata, TropicalSampleResult, TropicalSamplingSettings};

fn box_muller<T: MomTropFloat>(x1: &T, x2: &T) -> (T, T) {
    let r = (-x1.from_isize(2) * x1.ln()).sqrt();
    let theta = x1.from_isize(2) * x1.PI() * x2;
    (theta.cos() * &r, theta.sin() * &r)
}

#[derive(Debug, Clone, Copy)]
pub enum SamplingError {
    MatrixError(MatrixError),
}

pub fn sample<T: MomTropFloat, const D: usize, #[cfg(feature = "log")] L: Logger>(
    tropical_subgraph_table: &TropicalSubgraphTable,
    x_space_point: &[T],
    loop_signature: &[Vec<isize>],
    edge_data: &[(Option<T>, Vector<T, D>)],
    settings: &TropicalSamplingSettings,
    #[cfg(feature = "log")] logger: &L,
) -> Result<TropicalSampleResult<T, D>, SamplingError> {
    let num_loops = tropical_subgraph_table.tropical_graph.num_loops;

    let mut mimic_rng = MimicRng::new(x_space_point);
    let const_builder = mimic_rng.zero();

    let permatuhedral_sample = permatuhedral_sampling(
        tropical_subgraph_table,
        &mut mimic_rng,
        settings,
        #[cfg(feature = "log")]
        logger,
    );

    let l_matrix = compute_l_matrix(&permatuhedral_sample.x, loop_signature);
    let decomposed_l_matrix = match l_matrix.decompose_for_tropical(settings) {
        Ok(decomposition_result) => decomposition_result,
        Err(error) => {
            return Err(SamplingError::MatrixError(error));
        }
    };

    let lambda = inverse_gamma_lr(
        &const_builder.from_f64(tropical_subgraph_table.tropical_graph.dod),
        mimic_rng.get_random_number(Some("sample lambda")),
        50,
        &const_builder.from_f64(5.0),
    );

    if settings.print_debug_info {
        #[cfg(not(feature = "log"))]
        println!("lambda: {:?}", lambda);
        #[cfg(feature = "log")]
        logger.write("momtrop_lambda", &lambda.into())
    }

    let (edge_masses, edge_shifts): (Vec<T>, Vec<&Vector<T, D>>) = edge_data
        .iter()
        .map(|(option_mass, edge_shift)| {
            if let Some(mass) = option_mass {
                (mass.clone(), edge_shift)
            } else {
                (const_builder.zero(), edge_shift)
            }
        })
        .unzip();

    let q_vectors = sample_q_vectors(&mut mimic_rng, tropical_subgraph_table.dimension, num_loops);
    let u_vectors = compute_u_vectors(&permatuhedral_sample.x, loop_signature, &edge_shifts);

    let v_polynomial = compute_v_polynomial(
        &permatuhedral_sample.x,
        &u_vectors,
        &decomposed_l_matrix.inverse,
        &edge_shifts,
        &edge_masses,
    );

    let loop_momenta = compute_loop_momenta(
        &v_polynomial,
        &lambda,
        &decomposed_l_matrix.q_transposed_inverse,
        &q_vectors,
        &decomposed_l_matrix.inverse,
        &u_vectors,
    );

    if settings.print_debug_info {
        #[cfg(not(feature = "log"))]
        {
            println!("v: {:?}", &v_polynomial);
            println!("u: {:?}", &decomposed_l_matrix.determinant);
        }
        #[cfg(feature = "log")]
        {
            logger.write("momtrop_v", &v_polynomial.into());
            logger.write("momtrop_u", &decomposed_l_matrix.determinant.into())
        }
    }

    let u_trop = permatuhedral_sample.u_trop;
    let v_trop = permatuhedral_sample.v_trop;
    let u = &decomposed_l_matrix.determinant;
    let v = v_polynomial;

    let jacobian = (u_trop.ref_div(u))
        .powf(&const_builder.from_f64(tropical_subgraph_table.dimension as f64 / 2.0))
        * (v_trop.ref_div(&v))
            .powf(&const_builder.from_f64(tropical_subgraph_table.tropical_graph.dod))
        * const_builder.from_f64(tropical_subgraph_table.cached_factor);

    let metadata = if settings.return_metadata {
        Some(Metadata {
            l_matrix,
            q_vectors,
            lambda,
            shift: compute_only_shift(&decomposed_l_matrix.inverse, &u_vectors),
            decompoisiton_result: decomposed_l_matrix.clone(),
            u_vectors,
        })
    } else {
        None
    };

    Ok(TropicalSampleResult {
        loop_momenta,
        u_trop,
        v_trop,
        u: decomposed_l_matrix.determinant,
        v,
        jacobian,
        metadata,
    })
}

struct PermatuhedralSamplingResult<T: MomTropFloat> {
    x: Vec<T>,
    u_trop: T,
    v_trop: T,
}

/// This function returns the feynman parameters for a given graph and sample point, it also computes u_trop and v_trop.
/// A rescaling is performed for numerical stability, with this rescaling u_trop and v_trop always evaluate to 1.
#[inline]
fn permatuhedral_sampling<T: MomTropFloat, #[cfg(feature = "log")] L: Logger>(
    tropical_subgraph_table: &TropicalSubgraphTable,
    rng: &mut MimicRng<T>,
    settings: &TropicalSamplingSettings,
    #[cfg(feature = "log")] logger: &L,
) -> PermatuhedralSamplingResult<T> {
    let mut kappa = rng.one();
    let mut x_vec = vec![rng.zero(); tropical_subgraph_table.tropical_graph.topology.len()];
    let mut u_trop = rng.one();
    let mut v_trop = rng.one();

    let mut graph = tropical_subgraph_table
        .tropical_graph
        .get_full_subgraph_id();

    while !graph.is_empty() {
        // this saves a random variable
        let (edge, graph_without_edge) = if graph.has_one_edge() {
            let edge = graph
                .contains_edges()
                .next()
                .unwrap_or_else(|| unreachable!());
            let graph_without_edge = graph.pop_edge(edge);
            (edge, graph_without_edge)
        } else {
            tropical_subgraph_table.sample_edge(rng.get_random_number(Some("sample_edge")), &graph)
        };

        x_vec[edge] = kappa.clone();

        if tropical_subgraph_table.table[graph.get_id()].mass_momentum_spanning
            && !tropical_subgraph_table.table[graph_without_edge.get_id()].mass_momentum_spanning
        {
            v_trop = x_vec[edge].clone();
        }

        if tropical_subgraph_table.table[graph_without_edge.get_id()].loop_number
            < tropical_subgraph_table.table[graph.get_id()].loop_number
        {
            u_trop *= &x_vec[edge];
        }

        // Terminate early, so we do not waste a random variable in the final step
        graph = graph_without_edge;
        if graph.is_empty() {
            break;
        }

        let xi = rng.get_random_number(Some("sample xi"));
        kappa *= &xi.powf(
            &xi.from_f64(tropical_subgraph_table.table[graph.get_id()].generalized_dod)
                .inv(),
        );
    }

    let xi_trop = u_trop.ref_mul(&v_trop);

    // perform rescaling for numerical stability
    let target = u_trop.powf(&xi_trop.from_f64(-(tropical_subgraph_table.dimension as f64 / 2.0)))
        * (u_trop.ref_div(&xi_trop))
            .powf(&xi_trop.from_f64(tropical_subgraph_table.tropical_graph.dod));

    let loop_number = tropical_subgraph_table.table.last().unwrap().loop_number;
    let scaling = target.powf(
        &xi_trop
            .from_f64(
                tropical_subgraph_table.dimension as f64 / 2.0 * loop_number as f64
                    + tropical_subgraph_table.tropical_graph.dod,
            )
            .inv(),
    );

    if settings.print_debug_info {
        #[cfg(not(feature = "log"))]
        println!("feynman parameters before rescaling: {:?}", x_vec);
        #[cfg(feature = "log")]
        logger.write(
            "momtrop_feynman_parameter_no_rescaling",
            &x_vec.iter().map(|x| Into::<f64>::into(*x)).collect_vec(),
        );
    }

    x_vec.iter_mut().for_each(|x| *x *= &scaling);

    if settings.print_debug_info {
        #[cfg(not(feature = "log"))]
        {
            println!("sampled feynman parameters: {:?}", &x_vec);
            println!("u_trop before rescaling: {:?}", &u_trop);
            println!("v_trop before rescaling: {:?}", &v_trop);
        }
        #[cfg(feature = "log")]
        {
            logger.write(
                "momtrop_feynman_parameter",
                &x_vec.iter().map(|x| Into::<f64>::into(*x)).collect_vec(),
            );
            logger.write("momtrop_u_trop_no_rescaling", &u_trop.into());
            logger.write("momtrop_v_trop_no_rescaling", &v_trop.into());
        }
    }

    u_trop = u_trop.one();
    v_trop = v_trop.one();

    PermatuhedralSamplingResult {
        x: x_vec,
        u_trop,
        v_trop,
    }
}

/// Compute the L x L matrix from the feynman parameters and the signature matrix
#[inline]
fn compute_l_matrix<T: MomTropFloat>(
    x_vec: &[T],
    signature_matrix: &[Vec<isize>],
) -> SquareMatrix<T> {
    let num_edges = signature_matrix.len();
    let num_loops = signature_matrix[0].len();

    let mut temp_l_matrix = SquareMatrix::new_zeros_from_num(&x_vec[0], num_loops);

    for i in 0..num_loops {
        for j in i..num_loops {
            for e in 0..num_edges {
                let add = x_vec[e].from_isize(signature_matrix[e][i] * signature_matrix[e][j])
                    * &x_vec[e];
                if i == j {
                    temp_l_matrix[(i, j)] += &add;
                } else {
                    temp_l_matrix[(i, j)] += &add;
                    temp_l_matrix[(j, i)] += &add;
                }
            }
        }
    }

    temp_l_matrix
}

/// Sample Gaussian distributed vectors, using the Box-Muller transform
#[inline]
fn sample_q_vectors<T: MomTropFloat, const D: usize>(
    rng: &mut MimicRng<T>,
    dimension: usize,
    num_loops: usize,
) -> Vec<Vector<T, D>> {
    let token = Some("box muller");
    let num_variables = dimension * num_loops;

    let builder = rng.zero();

    let num_uniform_variables = num_variables + num_variables % 2;
    let mut gaussians = (0..num_uniform_variables / 2).flat_map(|_| {
        let (box_muller_1, box_muller_2) =
            box_muller(rng.get_random_number(token), rng.get_random_number(token));

        [box_muller_1, box_muller_2]
    });

    let mut res = Vec::with_capacity(num_loops);

    for _ in 0..num_loops {
        let mut vec = Vector::<T, D>::new_from_num(&builder);
        for i in 0..D {
            vec[i] = gaussians.next().unwrap_or_else(|| unreachable!());
        }
        res.push(vec);
    }

    res
}

/// Compute the vectors u, according to the formula in the notes
#[inline]
fn compute_u_vectors<T: MomTropFloat, const D: usize>(
    x_vec: &[T],
    signature_marix: &[Vec<isize>],
    edge_shifts: &[&Vector<T, D>],
) -> Vec<Vector<T, D>> {
    let num_loops = signature_marix[0].len();
    let num_edges = signature_marix.len();
    let const_builder = &x_vec[0];

    (0..num_loops)
        .map(|l| {
            (0..num_edges).fold(
                Vector::new_from_num(const_builder),
                |acc: Vector<T, D>, e| {
                    &acc + &(edge_shifts[e]
                        * (const_builder.from_isize(signature_marix[e][l]) * &x_vec[e]))
                },
            )
        })
        .collect_vec()
}

/// Compute the polynomial v, according to the formula in the notes
#[inline]
fn compute_v_polynomial<T: MomTropFloat, const D: usize>(
    x_vec: &[T],
    u_vectors: &[Vector<T, D>],
    inverse_l: &SquareMatrix<T>,
    edge_shifts: &[&Vector<T, D>],
    edge_masses: &[T],
) -> T {
    let num_loops = inverse_l.get_dim();
    let const_builder = &x_vec[0];

    let mut res = izip!(x_vec, edge_masses, edge_shifts)
        .map(|(x_e, mass, shift)| (mass.ref_mul(mass) + shift.squared()) * x_e)
        .fold(const_builder.zero(), |acc, x| acc + x);

    for l in 0..num_loops {
        res -= &(u_vectors[l].squared() * &inverse_l[(l, l)]);
    }

    for i in 0..num_loops {
        for j in i + 1..num_loops {
            res -= &(const_builder.from_isize(2)
                * u_vectors[i].dot(&u_vectors[j])
                * &inverse_l[(i, j)]);
        }
    }

    res
}

/// Compute the loop momenta, according to the formula in the notes
#[inline]
fn compute_loop_momenta<T: MomTropFloat, const D: usize>(
    v: &T,
    lambda: &T,
    q_t_inverse: &SquareMatrix<T>,
    q_vectors: &[Vector<T, D>],
    l_inverse: &SquareMatrix<T>,
    u_vectors: &[Vector<T, D>],
) -> Vec<Vector<T, D>> {
    let num_loops = q_t_inverse.get_dim();
    let prefactor = (v.ref_div(lambda) / lambda.from_isize(2)).sqrt();

    (0..num_loops)
        .map(|l| {
            q_vectors.iter().zip(u_vectors.iter()).enumerate().fold(
                q_vectors[0].new(),
                |acc, (l_prime, (q, u))| {
                    let q_part: Vector<T, D> = q * (prefactor.ref_mul(&q_t_inverse[(l, l_prime)]));
                    let u_part: Vector<T, D> = u * &l_inverse[(l, l_prime)];

                    &(&acc + &q_part) - &u_part
                },
            )
        })
        .collect_vec()
}

fn compute_only_shift<T: MomTropFloat, const D: usize>(
    l_inverse: &SquareMatrix<T>,
    u_vectors: &[Vector<T, D>],
) -> Vec<Vector<T, D>> {
    let num_loops = l_inverse.get_dim();

    (0..num_loops)
        .map(|l| {
            u_vectors.iter().enumerate().fold(
                Vector::new_from_num(&l_inverse.zero()),
                |acc, (l_prime, u)| {
                    let u_part: Vector<T, D> = u * &l_inverse[(l, l_prime)];
                    &acc + &u_part
                },
            )
        })
        .collect_vec()
}
