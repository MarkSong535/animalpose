//! CPU vs GPU (MPSGraph) forward of the general detector and pose networks.
//!
//! ```text
//! cargo run --release --features mps --example gpu_bench
//! ```

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    use animalpose::models::{load, Model};
    use animalpose::tensor::mpsgraph::GpuForward;
    use animalpose::tensor::{Ctx, Graph, NodeId};
    use std::time::Instant;

    let root = animalpose::data::repo_root().expect("run inside the repository");
    for (kind, shape) in [
        ("detector", vec![1, 3, 256, 320]),
        ("pose", vec![1, 3, 192, 192]),
        ("pose", vec![8, 3, 192, 192]),
    ] {
        let (cfg, k, w) = load(&root.join("weights").join(format!("{kind}_general.pt")))?;
        let model = Model::build(k, &cfg, &w)?;
        let n: usize = shape.iter().product();
        let x = vec![0.5f32; n];
        let forward = || {
            let mut g = Graph::new();
            let input = g.from_data(x.clone(), &shape);
            let out = model
                .forward_named(&mut Ctx::eval(), &mut g, &[input])
                .unwrap();
            (g, input, out)
        };
        let (g, input, out) = forward();
        let ids: Vec<NodeId> = out.iter().map(|o| o.1).collect();
        let t = Instant::now();
        let mut gpu = GpuForward::capture(&g, input, &ids)?;
        gpu.run(&x)?; // compile
        let compile = t.elapsed().as_secs_f64() * 1e3;
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            forward();
        }
        let cpu = t.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        let t = Instant::now();
        for _ in 0..reps {
            gpu.run(&x)?;
        }
        let mps = t.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        println!("{kind} {shape:?}: cpu {cpu:.1} ms, mps {mps:.1} ms ({:.1}x), capture+compile {compile:.0} ms", cpu / mps);
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("the MPSGraph benchmark requires macOS")
}
