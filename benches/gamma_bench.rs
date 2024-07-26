use criterion::{criterion_group, criterion_main, Criterion};
use momtrop::gamma::inverse_gamma_lr;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma sampling benchmarks");

    for omega in [0.5, 1.0, 2.0, 10.0] {
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            group.bench_function(&format!("omega = {omega}, p = {p}"), |b| {
                b.iter(|| inverse_gamma_lr(omega, p, 50, 5.0))
            });
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
