use criterion::{criterion_group, criterion_main, Criterion};
use momtrop::{vector::Vector, Edge, Graph, TropicalSamplingSettings};
use rand::{Rng, SeedableRng};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("graphs");

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
    let mut rng = rand::rngs::StdRng::seed_from_u64(69);
    let p1 = Vector::from_array([3.0, 4.0, 5.0]);
    let p2 = Vector::from_array([6.0, 7.0, 8.0]);
    let edge_data = vec![(None, Vector::new()), (None, p1), (None, (&p1 + &p2))];

    let x_space_point = vec![rng.r#gen(); sampler.get_dimension()];

    let settings = TropicalSamplingSettings::default();

    group.bench_function("triangle", |b| {
        b.iter(|| {
            sampler.generate_sample_from_x_space_point(&x_space_point, edge_data.clone(), &settings)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
