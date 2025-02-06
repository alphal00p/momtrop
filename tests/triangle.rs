use momtrop::{float::MomTropFloat, vector::Vector, Edge, Graph, TropicalSamplingSettings};
use rand::SeedableRng;

/// integrate a massless triangle with LTD and tropicalsampling
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
    let sampler = graph.build_sampler(loop_signature).unwrap();
    let p1 = Vector::from_array([3.0, 4.0, 5.0]);
    let p2 = Vector::from_array([6.0, 7.0, 8.0]);

    let mut rng = rand::rngs::StdRng::seed_from_u64(69);

    let mut sum = 0.0;

    let mut max_pol_ratio: f64 = 0.0;
    let mut min_pol_ratio: f64 = 1.0;

    let n_samples = 1000000;
    let p10 = 1.0;
    let p20 = 1.0;

    let edge_data = vec![
        (None, Vector::new_from_num(&p10)),
        (None, p1),
        (None, (&p1 + &p2)),
    ];

    let settings: TropicalSamplingSettings<()> = TropicalSamplingSettings {
        ..Default::default()
    };

    for _ in 0..n_samples {
        let sample = sampler
            .generate_sample_from_rng(edge_data.clone(), &settings, &mut rng)
            .unwrap();

        let energy_0 = energy_0(&sample.loop_momenta[0]);
        let energy_1 = energy_1(&sample.loop_momenta[0], &p1);
        let energy_2 = energy_2(&sample.loop_momenta[0], &p1, &p2);

        let energy_prefactor = energy_0.powf(2. * weight - 1.)
            * energy_1.powf(2. * weight - 1.)
            * energy_2.powf(2. * weight - 1.);
        let polynomial_ratio = (sample.u_trop / sample.u).powf(3. / 2.)
            * (sample.v_trop / sample.v).powf(sampler.get_dod());

        max_pol_ratio = max_pol_ratio.max(polynomial_ratio);
        min_pol_ratio = min_pol_ratio.min(polynomial_ratio);

        let pi_factor = (p10.from_isize(2) * p10.PI()).powf(p10.from_isize(3));
        let prefactor = energy_prefactor * sample.jacobian / pi_factor;

        let term1 = ((energy_0 + energy_1 + p10) * (energy_2 + energy_0 + p10 + p20)).inv();
        let term2 = ((energy_2 + energy_0 - p10 - p20) * (energy_1 + energy_2 - p20)).inv();
        let term3 = ((energy_0 + energy_1 + p10) * (energy_1 + energy_2 - p20)).inv();

        let term4 = ((energy_0 + energy_1 - p10) * (energy_2 + energy_0 - p10 - p20)).inv();
        let term5 = ((energy_2 + energy_0 + p10 + p20) * (energy_1 + energy_2 + p20)).inv();
        let term6 = ((energy_0 + energy_1 - p10) * (energy_1 + energy_2 + p20)).inv();

        sum += (term1 + term2 + term3 + term4 + term5 + term6) * prefactor;
    }

    let avg = sum / p10.from_isize(n_samples as isize);

    // theoretical bound
    assert!(
        min_pol_ratio
            >= (p10.one() / p10.from_isize(3))
                .powf(sampler.get_dimension() as f64 / 2. - sampler.get_dod())
                * (p10.one() / (p1.squared() + p2.squared() + (&p1 + &p2).squared()))
                    .powf(p10.from_f64(sampler.get_dod()))
    );

    assert!(max_pol_ratio <= (p10.one() / p1.squared()).powf(sampler.get_dod()));

    // this is the exact value with this seed, needs a more robust test
    assert_eq!(9.758362839019336e-5, avg);
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
