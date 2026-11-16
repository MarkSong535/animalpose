//! The plumbing every network needs: bare parameters as layers, and the tree walk that copies a
//! checkpoint into them.
//!
//! [`crate::tensor::Layer`] owns its tensors by checkpoint name, which is what makes a Python
//! state dict drop straight in. A few things in the Python models are not layers — `nn.Parameter`
//! arguments such as `PoseNet.heat_queries`, registered buffers such as `Normalize.mean`, and plain
//! `nn.ReLU` — so this module wraps them as [`Params`] and [`Relu`] leaves. A model then lists its
//! top-level layers through [`LayerTree`], and [`load_into`] copies a [`Weights`] file into every
//! slot of every layer, reporting the names that matched, the names the model wanted and did not get,
//! and the names it had no slot for.
//!
//! ```
//! use animalpose::models::layers::Params;
//! use animalpose::tensor::{Ctx, Graph, Layer};
//!
//! let p = Params::parameter("heat_queries", &[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! assert_eq!(p.params().join(","), "heat_queries");
//! let mut g = Graph::new();
//! let x = g.constant(0.0, &[1]);
//! let out = p.forward(&mut g, x, &mut Ctx::eval());
//! assert_eq!(g.shape(out).as_slice(), [2, 3]);
//! ```

use crate::models::weights::Weights;
use crate::tensor::{Ctx, Graph, Layer, NodeId, Slot};
use anyhow::Result;
use std::collections::BTreeSet;

/// A bare `nn.Parameter` or a registered buffer, wrapped as a [`Layer`] so the model tree collects it.
///
/// Mirrors the plain parameters of the Python models (`PoseNet.heat_queries`, `PoseNet.heat_bias`,
/// `PoseNet.off_queries`) and its registered buffers (`Normalize.mean`/`std`, `EgocentricFeatures.ant`
/// /`post`). [`Layer::forward`] ignores its input and pushes the tensor onto the tape, which is what the
/// network needs from a parameter.
pub struct Params {
    slots: Vec<Slot>,
    names: Vec<String>,
}

impl Params {
    /// A trainable parameter named `name`, initialised to `values` (a checkpoint load replaces them).
    pub fn parameter(name: &str, shape: &[usize], values: Vec<f32>) -> Self {
        Self::build(vec![slot(name, shape, values, true)])
    }

    /// A non-trainable buffer (`nn.Module.register_buffer`), e.g. the ImageNet normalisation constants.
    pub fn buffer(name: &str, shape: &[usize], values: Vec<f32>) -> Self {
        Self::build(vec![slot(name, shape, values, false)])
    }

    fn build(slots: Vec<Slot>) -> Self {
        let names = slots
            .iter()
            .filter(|s| s.trainable)
            .map(|s| s.name.clone())
            .collect();
        Self { slots, names }
    }

    /// The tensor's current values, row-major.
    pub fn values(&self) -> &[f32] {
        &self.slots[0].data
    }

    /// Overwrites the tensor, `false` when `values` has the wrong number of elements.
    pub fn set(&mut self, values: &[f32]) -> bool {
        if self.slots[0].data.len() != values.len() {
            return false;
        }
        self.slots[0].data.copy_from_slice(values);
        true
    }

    /// The tensor's shape, as stored in the checkpoint.
    pub fn shape(&self) -> &[usize] {
        &self.slots[0].shape
    }
}

/// One slot of a [`Params`] leaf.
fn slot(name: &str, shape: &[usize], values: Vec<f32>, trainable: bool) -> Slot {
    debug_assert_eq!(values.len(), Graph::numel(shape));
    Slot {
        name: name.to_string(),
        shape: shape.to_vec(),
        data: values,
        trainable,
        slot: 0,
    }
}

impl Layer for Params {
    fn forward(&self, g: &mut Graph, _x: NodeId, _ctx: &mut Ctx) -> NodeId {
        let s = &self.slots[0];
        g.param(s.slot, &s.shape, &s.data)
    }

    fn params(&self) -> &[String] {
        &self.names
    }

    fn visit(&self, f: &mut dyn FnMut(&Slot)) {
        for s in &self.slots {
            f(s);
        }
    }

    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Slot)) {
        for s in self.slots.iter_mut() {
            f(s);
        }
    }
}

/// `nn.ReLU` as a parameter-free [`Layer`], so an `nn.Sequential` keeps the checkpoint numbering the
/// Python module has (`embed.0`, `embed.1` = the ReLU, `embed.2`).
pub struct Relu;

impl Layer for Relu {
    fn forward(&self, g: &mut Graph, x: NodeId, _ctx: &mut Ctx) -> NodeId {
        g.relu(x)
    }

    fn params(&self) -> &[String] {
        &[]
    }

    fn visit(&self, _f: &mut dyn FnMut(&Slot)) {}

    fn visit_mut(&mut self, _f: &mut dyn FnMut(&mut Slot)) {}
}

/// The parameter tree of a network: the top-level layers a model is built from.
///
/// Every layer owns its children, so listing the top-level ones is enough to reach every slot; the
/// order is the checkpoint's (`nn.Module` registers submodules in assignment order).
pub trait LayerTree {
    /// Calls `f` with every top-level layer of this model.
    fn each_layer(&self, f: &mut dyn FnMut(&dyn Layer));

    /// Mutable counterpart of [`LayerTree::each_layer`].
    fn each_layer_mut(&mut self, f: &mut dyn FnMut(&mut dyn Layer));
}

/// What copying a [`Weights`] into a model did.
#[derive(Clone, Debug, Default)]
pub struct LoadReport {
    /// Checkpoint names that found a slot.
    pub loaded: Vec<String>,
    /// Slots the model has and the checkpoint did not provide.
    pub missing: Vec<String>,
    /// Checkpoint names with no slot, e.g. `*.num_batches_tracked`, which this port does not store.
    pub ignored: Vec<String>,
}

impl LoadReport {
    /// Errors when the checkpoint left a slot of the model unfilled.
    ///
    /// `// differs from Python:` `nn.Module.load_state_dict` also refuses unexpected keys; here they are
    /// only reported, because `num_batches_tracked` is deliberately not modelled.
    pub fn ensure_complete(&self) -> Result<()> {
        anyhow::ensure!(
            self.missing.is_empty(),
            "checkpoint does not provide {} tensor(s), first missing: {}",
            self.missing.len(),
            self.missing.join(", ")
        );
        Ok(())
    }
}

/// Copies every tensor of `weights` into the matching slot of `model`.
pub fn load_into<T: LayerTree + ?Sized>(model: &mut T, weights: &Weights) -> LoadReport {
    let wanted = slot_names(model);
    let mut report = LoadReport::default();
    for (name, tensor) in weights.tensors() {
        let mut hit = false;
        model.each_layer_mut(&mut |layer| {
            if layer.load(name, &tensor.data) {
                hit = true;
            }
        });
        if hit {
            report.loaded.push(name.clone());
        } else {
            report.ignored.push(name.clone());
        }
    }
    report.missing = wanted
        .into_iter()
        .filter(|name| weights.get(name).is_none())
        .collect();
    report
}

/// Rewrites the slot names of `layer` through `rename`, which returns `None` to leave a name alone;
/// returns how many slots changed.
///
/// The tape's builders number the steps of a `nn.Sequential` (`{name}.0.weight`, `{name}.1.bias`),
/// which is what most Python modules do too. Where a module names its submodules instead — torchvision's
/// `conv1` / `bn1` / `downsample.0` — the layer is built the same way and renamed here, so the
/// checkpoint names still match the Python state dict exactly.
pub fn rename_slots<L: Layer + ?Sized>(
    layer: &mut L,
    rename: &dyn Fn(&str) -> Option<String>,
) -> usize {
    let mut hit = 0;
    layer.visit_mut(&mut |slot| {
        if let Some(name) = rename(&slot.name) {
            slot.name = name;
            hit += 1;
        }
    });
    hit
}

/// Numbers every trainable slot of `model` from `base`, so [`crate::tensor::Graph::param_grads`]
/// can hand a training step the gradients in the model's own parameter order.
pub fn assign_slots<T: LayerTree + ?Sized>(model: &mut T, base: usize) -> usize {
    let mut next = base;
    model.each_layer_mut(&mut |layer| {
        next = layer.assign_slots(next);
    });
    next
}

/// The trainable slot names of `model`, in slot order (the order a checkpoint stores them in).
pub fn param_names<T: LayerTree + ?Sized>(model: &T) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut names = Vec::new();
    model.each_layer(&mut |layer| {
        layer.visit(&mut |slot| {
            if slot.trainable && seen.insert(slot.name.clone()) {
                names.push(slot.name.clone());
            }
        });
    });
    names
}

/// Every checkpoint name `model` has a slot for, trainable parameters and buffers alike.
pub fn slot_names<T: LayerTree + ?Sized>(model: &T) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut names = Vec::new();
    model.each_layer(&mut |layer| {
        layer.visit(&mut |slot| {
            if seen.insert(slot.name.clone()) {
                names.push(slot.name.clone());
            }
        });
    });
    names
}
