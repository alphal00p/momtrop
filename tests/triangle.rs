use momtrop::{vector::Vector, Edge, Graph};
use rand::SeedableRng;

#[test]
fn integrate_massless_triangle() {
    let weight = 2. / 3.;

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

    let externals = vec![0, 1, 2];

    let graph = Graph {
        edges: triangle_edges,
        externals,
    };

    let loop_signature = vec![vec![1]; 3];
    let sampler = graph.build_sampler(loop_signature, 3).unwrap();
    let p1 = Vector::from_array([3.0, 4.0, 5.0]);
    let p2 = Vector::from_array([6.0, 7.0, 8.0]);

    let mut rng = rand::rngs::StdRng::seed_from_u64(69);

    let mut sum = 0.0;

    let mut max_pol_ratio: f64 = 0.0;
    let mut min_pol_ratio: f64 = 1.0;

    let n_samples = 1000000;

    let edge_data = vec![
        (None, Vector::new()),
        (None, p1.clone()),
        (None, (&p1 + &p2).clone()),
    ];

    let p10 = 1.0;
    let p20 = 1.0;

    for _ in 0..n_samples {
        let sample = sampler.generate_sample_from_rng(edge_data.clone(), false, &mut rng);

        let energy_0 = energy_0(&sample.loop_momenta[0]);
        let energy_1 = energy_1(&sample.loop_momenta[0], &p1);
        let energy_2 = energy_2(&sample.loop_momenta[0], &p1, &p2);

        let energy_prefactor = energy_0.powf(2. * weight - 1.)
            * energy_1.powf(2. * weight - 1.)
            * energy_2.powf(2. * weight - 1.);
        let pi_prefactor = std::f64::consts::PI.powf(-1.5);
        let factor_2_prefactor = 1. / 64.;
        let polynomial_ratio = (sample.u_trop / sample.u).powf(3. / 2.)
            * (sample.v_trop / sample.v).powf(sampler.get_dod());

        max_pol_ratio = max_pol_ratio.max(polynomial_ratio);
        min_pol_ratio = min_pol_ratio.min(polynomial_ratio);

        let prefactor = energy_prefactor
            * pi_prefactor
            * factor_2_prefactor
            * polynomial_ratio
            * sample.prefactor;

        let term1 = ((energy_0 + energy_1 + p10) * (energy_2 + energy_0 + p10 + p20)).recip();
        let term2 = ((energy_2 + energy_0 - p10 - p20) * (energy_1 + energy_2 - p20)).recip();
        let term3 = ((energy_0 + energy_1 + p10) * (energy_1 + energy_2 - p20)).recip();

        let term4 = ((energy_0 + energy_1 - p10) * (energy_2 + energy_0 - p10 - p20)).recip();
        let term5 = ((energy_2 + energy_0 + p10 + p20) * (energy_1 + energy_2 + p20)).recip();
        let term6 = ((energy_0 + energy_1 - p10) * (energy_1 + energy_2 + p20)).recip();

        sum += (term1 + term2 + term3 + term4 + term5 + term6) * prefactor;
    }

    let avg = sum / n_samples as f64;

    // theoretical bound
    assert!(
        min_pol_ratio
            >= (1f64 / 3f64).powf(sampler.get_dimension() as f64 / 2. - sampler.get_dod())
                * (1f64 / (p1.squared() + p2.squared() + (&p1 + &p2).squared()))
                    .powf(sampler.get_dod())
    );

    assert!(max_pol_ratio <= (1f64 / p1.squared()).powf(sampler.get_dod()));

    // this is the exact value with this seed, needs a more robust test
    assert_eq!(9.758362839019333e-5, avg);
}

fn energy_0(k: &Vector<f64, 3>) -> f64 {
    k.squared().sqrt()
}

fn energy_1(k: &Vector<f64, 3>, p: &Vector<f64, 3>) -> f64 {
    (k + p).squared().sqrt()
}

fn energy_2(k: &Vector<f64, 3>, p1: &Vector<f64, 3>, p2: &Vector<f64, 3>) -> f64 {
    (k + &(p1 + p2)).squared().sqrt()
}
