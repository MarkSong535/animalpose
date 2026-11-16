//! Optimisation: `torch.optim.AdamW` plus the warmup-and-cosine schedule the Python trainers use.
//!
//! `train/loops.py:_optimizer` builds `AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)` (betas
//! `(0.9, 0.999)`, `eps = 1e-8`, PyTorch's defaults) and a `LambdaLR` multiplying the learning rate by
//! `min(1, (s + 1) / warm) * 0.5 * (1 + cos(pi * min(s, steps) / steps))` with
//! `warm = max(1, int(0.05 * steps))`. [`Schedule`] is that multiplier and [`AdamW`] is the update.

/// `true` when `name` takes weight decay: everything except biases and normalisation parameters.
///
/// `// differs from Python:` plain `torch.optim.AdamW` decays every tensor it is given; animalpose's
/// checkpoints keep biases and norm gains at a fixed scale, so those names are exempt here (the
/// `Layer::params()` names of `nn.rs` are what is inspected: `*.bias`, `*.running_*`, `norm`, `gamma`,
/// `beta`).
pub fn decays(name: &str) -> bool {
    let lower = name.to_lowercase();
    !["bias", "norm", "gamma", "beta"]
        .iter()
        .any(|key| lower.contains(key))
}

/// `torch.optim.AdamW` with the animalpose defaults, holding one moment estimate per parameter.
#[derive(Clone, Debug)]
pub struct AdamW {
    /// Base learning rate (`train.lr`, `1e-3` in the configs).
    pub lr: f32,
    /// Decoupled weight decay (`train.weight_decay`, `1e-4` in the configs).
    pub weight_decay: f32,
    /// `betas` of `torch.optim.AdamW`.
    pub betas: (f32, f32),
    /// `eps` of `torch.optim.AdamW`.
    pub eps: f32,
    /// Multiplier of `lr` for the next update: the [`Schedule`] value at the current step.
    pub scale: f32,
    step: usize,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Default for AdamW {
    fn default() -> Self {
        Self::new(1e-3, 1e-4)
    }
}

impl AdamW {
    /// An optimiser with the given base learning rate and weight decay.
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            weight_decay,
            betas: (0.9, 0.999),
            eps: 1e-8,
            scale: 1.0,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Number of updates applied so far (PyTorch's `state['step']`).
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Sets the schedule multiplier used by the next [`AdamW::step`].
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// `lr * scale`, the learning rate the next update uses.
    pub fn current_lr(&self) -> f32 {
        self.lr * self.scale
    }

    /// Forgets the moment estimates, e.g. when the parameter list changes size.
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
    }

    /// One update over the whole parameter list; see [`adamw_step`].
    pub fn step(&mut self, params: &mut [Vec<f32>], grads: &[Vec<f32>], names: &[String]) {
        adamw_step(params, grads, names, self);
    }

    /// Resizes the moment buffers to the current parameter list.
    fn fit(&mut self, params: &[Vec<f32>]) {
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0f32; p.len()]).collect();
            self.v = params.iter().map(|p| vec![0.0f32; p.len()]).collect();
            return;
        }
        for (buf, p) in self.m.iter_mut().zip(params) {
            buf.resize(p.len(), 0.0);
        }
        for (buf, p) in self.v.iter_mut().zip(params) {
            buf.resize(p.len(), 0.0);
        }
    }
}

/// One `torch.optim.AdamW` update: bias-corrected moments plus decoupled weight decay, skipped for the
/// names [`decays`] rejects.
///
/// `params` (live values), `grads` and `names` are parallel lists, in the order
/// [`crate::tensor::Layer::params`] produces; the moment estimates live in `opt`, whose `scale` is the current
/// [`Schedule`] value. A missing gradient (an empty or short slice) counts as zero, like
/// `param.grad = None`.
pub fn adamw_step(params: &mut [Vec<f32>], grads: &[Vec<f32>], names: &[String], opt: &mut AdamW) {
    assert_eq!(
        params.len(),
        grads.len(),
        "adamw has {} params but {} grads",
        params.len(),
        grads.len()
    );
    opt.fit(params);
    opt.step += 1;
    let (b1, b2) = opt.betas;
    let lr = opt.current_lr();
    let t = opt.step as f32;
    let bias1 = 1.0 - b1.powf(t);
    let bias2 = 1.0 - b2.powf(t);
    let step_size = lr / bias1;
    let inv_bias2 = 1.0 / bias2.sqrt();
    let decay = if opt.weight_decay != 0.0 {
        opt.weight_decay * lr
    } else {
        0.0
    };
    let eps = opt.eps;
    for (i, p) in params.iter_mut().enumerate() {
        let grad = &grads[i];
        let name = names.get(i).map(String::as_str).unwrap_or("");
        let decay = if decays(name) { decay } else { 0.0 };
        let (m, v) = (&mut opt.m[i], &mut opt.v[i]);
        for (j, w) in p.iter_mut().enumerate() {
            let g = grad.get(j).copied().unwrap_or(0.0);
            m[j] = b1 * m[j] + (1.0 - b1) * g;
            v[j] = b2 * v[j] + (1.0 - b2) * g * g;
            let denom = v[j].sqrt() * inv_bias2 + eps;
            *w = *w * (1.0 - decay) - step_size * m[j] / denom;
        }
    }
}

/// The trainers' warmup-plus-cosine multiplier of `train/loops.py:_optimizer`.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    /// Total number of steps the cosine decay spans.
    pub steps: usize,
    /// Length of the linear warmup, `max(1, int(0.05 * steps))` by default.
    pub warmup: usize,
}

impl Schedule {
    /// The Python schedule for `steps` training steps.
    pub fn new(steps: usize) -> Self {
        Self::with_warmup(steps, 0.05)
    }

    /// The same shape with an explicit warmup *fraction* (the Python code uses `0.05`).
    pub fn with_warmup(steps: usize, warm_fraction: f32) -> Self {
        let steps = steps.max(1);
        Self {
            steps,
            warmup: ((steps as f32 * warm_fraction) as usize).max(1),
        }
    }

    /// The multiplier at step `s`: `min(1, (s + 1) / warm) * 0.5 * (1 + cos(pi * min(s, steps) / steps))`.
    pub fn at(&self, step: usize) -> f32 {
        let warm = self.warmup.max(1) as f32;
        let s = step as f32;
        let warmup = (1.0 + s).min(warm) / warm;
        let progress = s.min(self.steps as f32) / self.steps as f32;
        warmup * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}
