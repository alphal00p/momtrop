use std::f64::consts::PI;

// This is a full example showcasing how to use momtrop.
// Import the necessary structs
use momtrop::{
    Edge, Graph, TropicalSamplingSettings, assert_approx_eq, float::MomTropFloat, vector::Vector,
};

// We also need a random number generator, a seedable RNG is used for reproducibility
use rand::SeedableRng;

/// integrate a massless triangle with LTD and tropicalsampling
#[test]
fn integrate_massless_triangle() {
    // We choose a weight nu_e = 2/3 for each edge of the triangle.
    let weight = 2. / 3.;

    // Define the edges of the triangle, each edge is massless and has the same weight
    let triangle_edges = vec![
        Edge {
            vertices: (0, 1),
            is_massive: false,
            weight,
        },
        Edge {
            vertices: (1, 2),
            is_massive: false,
            weight,
        },
        Edge {
            vertices: (2, 0),
            is_massive: false,
            weight,
        },
    ];

    // Here we define which vertices have an external momentum flowing in. In this case it is all three vertices.
    let externals = vec![0, 1, 2];

    let graph = Graph {
        edges: triangle_edges,
        externals,
    };

    // Here we define the signature matrix, this is the only choice for this example.
    let loop_signature = vec![vec![1]; 3];

    // .build_sampler() construct the sampler for the given graph and signature matrix.
    let sampler = graph.build_sampler(loop_signature).unwrap();

    // We can use the Vector struct to represent our external momenta.
    let p1 = Vector::from_array([3.0, 4.0, 5.0]);
    let p2 = Vector::from_array([6.0, 7.0, 8.0]);

    // setup a random number generator
    let mut rng = rand::rngs::StdRng::seed_from_u64(420);

    // keep track of the value of the integral.
    let mut sum = 0.0;

    let mut sq_sum = 0.0;

    let n_samples = 1000000;

    // Specify the energy compoents of the external momenta.
    let p10 = 1.0;
    let p20 = 1.0;

    // Here we compute the masses and shifts of each edge. Since it is massless, we use None.
    let edge_data = vec![
        (None, Vector::from_array([0.0, 0.0, 0.0])), // We choose the first edge to be the loop momentum basis, so the shift is zero.
        (None, p1),
        (None, (&p1 + &p2)),
    ];

    // default settings
    let settings = TropicalSamplingSettings {
        ..Default::default()
    };

    // This is the number of random variable that we need
    // let dimension = sampler.get_dimension();

    // start the integration loop
    for _ in 0..n_samples {
        // generate the random variables for the sample.
        //let random_variables = std::iter::repeat_with(|| rng.r#gen::<f64>())
        //    .take(dimension)
        //    .collect::<Vec<_>>();

        // generate a sample point
        let sample = sampler
            .generate_sample_from_rng(edge_data.clone(), &settings, &mut rng)
            .unwrap();

        // Implementation of the integrand

        // we first compute the onshell energies
        let energy_0 = energy_0(&sample.loop_momenta[0]);
        let energy_1 = energy_1(&sample.loop_momenta[0], &p1);
        let energy_2 = energy_2(&sample.loop_momenta[0], &p1, &p2);

        // We compute the 1/E prefactor of CFF, we adjust the power to compensate for the higher weight in the graph provided to the sampler.
        let energy_prefactor = energy_0.powf(2. * weight - 1.)
            * energy_1.powf(2. * weight - 1.)
            * energy_2.powf(2. * weight - 1.);

        let pi_factor = (2. * PI).powf(3.0); // we need to include the the factor (2*pi)^3 for the spatial loop momentum integration.
        let prefactor = energy_prefactor * sample.jacobian / pi_factor / 8.0; // we add a factor 1/2 for each edge

        // now we compute the 6 CFF terms of the triangle
        let term1 = ((energy_0 + energy_1 + p10) * (energy_2 + energy_0 + p10 + p20)).inv();
        let term2 = ((energy_2 + energy_0 - p10 - p20) * (energy_1 + energy_2 - p20)).inv();
        let term3 = ((energy_0 + energy_1 + p10) * (energy_1 + energy_2 - p20)).inv();

        let term4 = ((energy_0 + energy_1 - p10) * (energy_2 + energy_0 - p10 - p20)).inv();
        let term5 = ((energy_2 + energy_0 + p10 + p20) * (energy_1 + energy_2 + p20)).inv();
        let term6 = ((energy_0 + energy_1 - p10) * (energy_1 + energy_2 + p20)).inv();

        // The integrand is the sum of the 6 terms multiplied by the prefactor.
        let value = (term1 + term2 + term3 + term4 + term5 + term6) * prefactor;

        sum += value;
        sq_sum += value * value;
    }

    // compute the average to get an estimate of the integral
    let avg = sum / n_samples as f64;
    let err = ((sq_sum / n_samples as f64 - avg * avg) / n_samples as f64).sqrt();

    // If everything is correct, the result should be approximately 9.76-5
    assert_approx_eq(&avg, &9.758362839019336e-5, &1.0e-2);

    println!();
    println!("integral: {}", avg);
    println!("error:    {}", err);
    println!("target:   {}", 0.0000976546);
    println!();
}

// helper functions to compute the onshell energies
fn energy_0(k: &Vector<f64, 3>) -> f64 {
    k.squared().sqrt()
}

fn energy_1(k: &Vector<f64, 3>, p: &Vector<f64, 3>) -> f64 {
    (k + p).squared().sqrt()
}

fn energy_2(k: &Vector<f64, 3>, p1: &Vector<f64, 3>, p2: &Vector<f64, 3>) -> f64 {
    (k + &(p1 + p2)).squared().sqrt()
}
