use momtrop::{
    vector::Vector, Edge, Graph, SampleGenerator, TropicalSampleResult, TropicalSamplingSettings,
};

#[test]
fn test_prec_eq() {
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
    let sampler: SampleGenerator<3> = graph.build_sampler(loop_signature, 3).unwrap();
    let p1 = Vector::from_array([3.0, 4.0, 5.0]);
    let p2 = Vector::from_array([6.0, 7.0, 8.0]);

    let edge_data = vec![
        (Option::<f64>::None, Vector::new()),
        (None, p1),
        (None, (&p1 + &p2)),
    ];

    let x_space_point = vec![0.1; sampler.get_dimension()];
    let settings = TropicalSamplingSettings::default();

    let sample = sampler
        .generate_sample_from_x_space_point(&x_space_point, edge_data.clone(), &settings)
        .unwrap();
    let sample_f128 = sampler
        .generate_sample_f128_from_x_space_point(&x_space_point, edge_data.clone(), &settings)
        .unwrap()
        .downcast();

    assert_approx_eq_sample(sample, sample_f128, 1.0e-15);
}

fn assert_approx_eq_sample<const D: usize>(
    sample_1: TropicalSampleResult<f64, D>,
    sample_2: TropicalSampleResult<f64, D>,
    tolerance: f64,
) {
    for (loop_mom_1, loop_mom_2) in sample_1
        .loop_momenta
        .iter()
        .zip(sample_2.loop_momenta.iter())
    {
        let elements_1 = loop_mom_1.get_elements();
        let elements_2 = loop_mom_2.get_elements();

        for (&el_1, &el_2) in elements_1.iter().zip(elements_2.iter()) {
            assert_approx_eq(el_1, el_2, tolerance);
        }
    }

    assert_approx_eq(sample_1.u_trop, sample_2.u_trop, tolerance);
    assert_approx_eq(sample_1.v_trop, sample_2.v_trop, tolerance);
    assert_approx_eq(sample_1.u, sample_2.u, tolerance);
    assert_approx_eq(sample_1.v, sample_2.v, tolerance);
    assert_approx_eq(sample_1.jacobian, sample_2.jacobian, tolerance);
}

fn assert_approx_eq(x: f64, y: f64, tolerance: f64) {
    let avg = (x + y) / 2.;
    let norm_err = ((x - y) / avg).abs();
    assert!(norm_err < tolerance);
}
