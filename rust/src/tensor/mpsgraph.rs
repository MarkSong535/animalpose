//! Inference on the GPU: a recorded forward pass replayed as one Metal Performance Shaders graph.
//!
//! The tape of [`crate::tensor::Graph`] evaluates eagerly, so one CPU forward pass leaves behind every
//! operation, every shape and every weight. [`GpuForward::capture`] translates that tape into an
//! `MPSGraph`: the network input becomes a placeholder, weights become constants that stay in GPU
//! memory, and batch norm in eval mode folds into a per-channel scale and shift. [`GpuForward::run`]
//! then feeds a new input of the same shape and reads the outputs back. Only the inference
//! operations are translated; a tape holding anything else (an LSTM, a loss) is refused, and the
//! caller keeps the CPU path.
//!
//! After every translated operation the result is reshaped to the shape the CPU tape recorded, so the
//! two paths agree on layouts by construction.

use std::collections::HashMap;
use std::ptr::NonNull;

use anyhow::{bail, ensure, Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::{NSArray, NSData, NSDictionary, NSNumber};
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphConvolution2DOpDescriptor, MPSGraphDevice, MPSGraphPaddingStyle,
    MPSGraphPooling2DOpDescriptor, MPSGraphTensor, MPSGraphTensorData,
    MPSGraphTensorNamedDataLayout,
};
use objc2_metal_performance_shaders::MPSShape;

use super::ops::BnMode;
use super::{Graph, NodeId, Op};

type Tensor = Retained<MPSGraphTensor>;

/// A captured forward pass, compiled for one input shape.
pub struct GpuForward {
    graph: Retained<MPSGraph>,
    device: Retained<MPSGraphDevice>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    input: Tensor,
    input_shape: Vec<usize>,
    outputs: Vec<(Tensor, Vec<usize>)>,
}

// SAFETY: MPSGraph, its tensors, the Metal device and command queue may be used from any thread as
// long as one graph is not run concurrently, which `&mut self` in `run` and the callers' mutexes ensure.
unsafe impl Send for GpuForward {}
unsafe impl Sync for GpuForward {}

fn shape_of(dims: &[usize]) -> Retained<MPSShape> {
    let numbers: Vec<Retained<NSNumber>> = dims
        .iter()
        .map(|d| NSNumber::new_usize(*d))
        .collect();
    NSArray::from_retained_slice(&numbers)
}

fn f32_bytes(data: &[f32]) -> Retained<NSData> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    NSData::with_bytes(&bytes)
}

impl GpuForward {
    /// Translates the part of `g` that `outputs` depend on, with `input` as the fed placeholder.
    pub fn capture(g: &Graph, input: NodeId, outputs: &[NodeId]) -> Result<Self> {
        let mtl = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = mtl.newCommandQueue().context("no Metal command queue")?;
        let device = unsafe { MPSGraphDevice::deviceWithMTLDevice(&mtl) };
        let graph = unsafe { MPSGraph::new() };

        // Only the nodes the outputs depend on are translated.
        let mut needed = vec![false; g.nodes.len()];
        let mut stack: Vec<usize> = outputs.iter().map(|o| o.0).collect();
        while let Some(i) = stack.pop() {
            if needed[i] {
                continue;
            }
            needed[i] = true;
            if i != input.0 {
                stack.extend(inputs_of(&g.nodes[i].op).iter().map(|n| n.0));
            }
        }

        let input_shape = g.nodes[input.0].shape.clone();
        let placeholder = unsafe {
            graph.placeholderWithShape_dataType_name(
                Some(&shape_of(&input_shape)),
                MPSDataType::Float32,
                None,
            )
        };
        let mut built: HashMap<usize, Tensor> = HashMap::new();
        built.insert(input.0, placeholder.clone());
        for (i, &need) in needed.iter().enumerate() {
            if !need || i == input.0 {
                continue;
            }
            let t = translate(&graph, g, i, &built)
                .with_context(|| format!("node {i} ({})", op_name(&g.nodes[i].op)))?;
            let t = reshape(&graph, &t, &g.nodes[i].shape);
            built.insert(i, t);
        }
        let outputs = outputs
            .iter()
            .map(|o| (built[&o.0].clone(), g.nodes[o.0].shape.clone()))
            .collect();
        Ok(Self {
            graph,
            device,
            queue,
            input: placeholder,
            input_shape,
            outputs,
        })
    }

    /// The input shape this capture was compiled for.
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Runs the captured pass on `input` (the captured input shape, row-major) and returns every
    /// output, row-major, in the order given to [`Self::capture`].
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<Vec<f32>>> {
        let n: usize = self.input_shape.iter().product();
        ensure!(input.len() == n, "GPU input holds {} values, the capture wants {n}", input.len());
        unsafe {
            let data = MPSGraphTensorData::initWithDevice_data_shape_dataType(
                MPSGraphTensorData::alloc(),
                &self.device,
                &f32_bytes(input),
                &shape_of(&self.input_shape),
                MPSDataType::Float32,
            );
            let feeds = NSDictionary::from_retained_objects(&[&*self.input], &[data]);
            let targets: Vec<Tensor> = self.outputs.iter().map(|o| o.0.clone()).collect();
            let results = self.graph.runWithMTLCommandQueue_feeds_targetTensors_targetOperations(
                &self.queue,
                &feeds,
                &NSArray::from_retained_slice(&targets),
                None,
            );
            let mut out = Vec::with_capacity(self.outputs.len());
            for (tensor, shape) in &self.outputs {
                let data = results
                    .objectForKey(tensor)
                    .context("the GPU run returned no value for an output")?;
                let len: usize = shape.iter().product();
                let mut values = vec![0f32; len.max(1)];
                data.mpsndarray().readBytes_strideBytes(
                    NonNull::new(values.as_mut_ptr().cast()).context("null output buffer")?,
                    std::ptr::null_mut(),
                );
                values.truncate(len);
                out.push(values);
            }
            Ok(out)
        }
    }
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Constant => "constant",
        Op::Param(_) => "param",
        Op::Add(..) => "add",
        Op::Sub(..) => "sub",
        Op::Mul(..) => "mul",
        Op::Scale(..) => "scale",
        Op::Relu(_) => "relu",
        Op::Sigmoid(_) => "sigmoid",
        Op::Tanh(_) => "tanh",
        Op::Softmax(..) => "softmax",
        Op::LogSoftmax(..) => "log_softmax",
        Op::Matmul(..) => "matmul",
        Op::Conv2d { .. } => "conv2d",
        Op::MaxPool2d { .. } => "max_pool2d",
        Op::AvgPool2d { .. } => "avg_pool2d",
        Op::UpsampleNearest { .. } => "upsample_nearest",
        Op::Reshape(..) => "reshape",
        Op::Permute(..) => "permute",
        Op::Concat(..) => "concat",
        Op::Slice { .. } => "slice",
        Op::ReduceSum(..) => "reduce_sum",
        Op::ReduceMean(..) => "reduce_mean",
        Op::BatchNorm { .. } => "batch_norm",
        Op::L2Normalize(..) => "l2_normalize",
        Op::BceWithLogits { .. } => "bce_with_logits",
        Op::WeightedLoss { .. } => "weighted_loss",
        Op::CrossEntropy { .. } => "cross_entropy",
        Op::Lstm { .. } => "lstm",
        Op::Dropout { .. } => "dropout",
    }
}

/// The nodes an operation reads.
fn inputs_of(op: &Op) -> Vec<NodeId> {
    match op {
        Op::Constant | Op::Param(_) => vec![],
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Matmul(a, b) => vec![*a, *b],
        Op::Scale(x, _)
        | Op::Relu(x)
        | Op::Sigmoid(x)
        | Op::Tanh(x)
        | Op::Softmax(x, _)
        | Op::LogSoftmax(x, _)
        | Op::Reshape(x, _)
        | Op::Permute(x, _)
        | Op::ReduceSum(x, _)
        | Op::ReduceMean(x, _)
        | Op::L2Normalize(x, _)
        | Op::MaxPool2d { x, .. }
        | Op::AvgPool2d { x, .. }
        | Op::UpsampleNearest { x, .. }
        | Op::Slice { x, .. }
        | Op::Dropout { x, .. } => vec![*x],
        Op::Conv2d { x, w, b, .. } => {
            let mut v = vec![*x, *w];
            v.extend(b);
            v
        }
        Op::Concat(xs, _) => xs.clone(),
        Op::BatchNorm { x, .. } => vec![*x],
        Op::BceWithLogits { logits, target } => vec![*logits, *target],
        Op::WeightedLoss { x, .. } => vec![*x],
        Op::CrossEntropy {
            logits,
            target,
            weight,
            ..
        } => {
            let mut v = vec![*logits, *target];
            v.extend(weight);
            v
        }
        Op::Lstm { x, w, .. } => {
            let mut v = vec![*x];
            v.extend(w);
            v
        }
    }
}

fn constant(graph: &MPSGraph, data: &[f32], shape: &[usize]) -> Tensor {
    unsafe {
        graph.constantWithData_shape_dataType(&f32_bytes(data), &shape_of(shape), MPSDataType::Float32)
    }
}

fn scalar(graph: &MPSGraph, v: f64) -> Tensor {
    unsafe { graph.constantWithScalar_dataType(v, MPSDataType::Float32) }
}

fn reshape(graph: &MPSGraph, t: &MPSGraphTensor, shape: &[usize]) -> Tensor {
    unsafe { graph.reshapeTensor_withShape_name(t, &shape_of(shape), None) }
}

fn axes(list: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = list.iter().map(|a| NSNumber::new_usize(*a)).collect();
    NSArray::from_retained_slice(&numbers)
}

/// One tape node as an MPSGraph tensor.
fn translate(graph: &MPSGraph, g: &Graph, i: usize, built: &HashMap<usize, Tensor>) -> Result<Tensor> {
    let node = &g.nodes[i];
    let get = |n: &NodeId| -> Result<Tensor> {
        built.get(&n.0).cloned().with_context(|| format!("node {} was not translated", n.0))
    };
    let shape = |n: &NodeId| g.nodes[n.0].shape.clone();
    unsafe {
        Ok(match &node.op {
            Op::Constant | Op::Param(_) => constant(graph, &node.data, &node.shape),
            Op::Add(a, b) => graph.additionWithPrimaryTensor_secondaryTensor_name(&*get(a)?, &*get(b)?, None),
            Op::Sub(a, b) => graph.subtractionWithPrimaryTensor_secondaryTensor_name(&*get(a)?, &*get(b)?, None),
            Op::Mul(a, b) => graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&*get(a)?, &*get(b)?, None),
            Op::Scale(x, f) => graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                &*get(x)?,
                &scalar(graph, f64::from(*f)),
                None,
            ),
            Op::Relu(x) => graph.reLUWithTensor_name(&*get(x)?, None),
            Op::Sigmoid(x) => graph.sigmoidWithTensor_name(&*get(x)?, None),
            Op::Tanh(x) => graph.tanhWithTensor_name(&*get(x)?, None),
            Op::Softmax(x, axis) => graph.softMaxWithTensor_axis_name(&*get(x)?, *axis as isize, None),
            Op::LogSoftmax(x, axis) => {
                let s = graph.softMaxWithTensor_axis_name(&*get(x)?, *axis as isize, None);
                graph.logarithmWithTensor_name(&s, None)
            }
            Op::Matmul(a, b) => {
                let (sa, sb) = (shape(a), shape(b));
                let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
                let n = sb[sb.len() - 1];
                let ba: usize = sa[..sa.len() - 2].iter().product();
                let bb: usize = sb[..sb.len() - 2].iter().product();
                let a3 = reshape(graph, &*get(a)?, &[ba.max(1), m, k]);
                let b3 = reshape(graph, &*get(b)?, &[bb.max(1), k, n]);
                graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(&a3, &b3, None)
            }
            Op::Conv2d {
                x,
                w,
                b,
                stride,
                pad,
                dilation,
            } => {
                let desc = MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(
                    *stride, *stride, *dilation, *dilation, 1, *pad, *pad, *pad, *pad,
                    MPSGraphPaddingStyle::Explicit,
                    MPSGraphTensorNamedDataLayout::NCHW,
                    MPSGraphTensorNamedDataLayout::OIHW,
                )
                .context("invalid convolution descriptor")?;
                let y = graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                    &*get(x)?,
                    &*get(w)?,
                    &desc,
                    None,
                );
                match b {
                    Some(b) => {
                        let c = shape(b).iter().product::<usize>();
                        let bias = reshape(graph, &*get(b)?, &[1, c, 1, 1]);
                        graph.additionWithPrimaryTensor_secondaryTensor_name(&y, &bias, None)
                    }
                    None => y,
                }
            }
            Op::MaxPool2d { x, k, stride, pad } | Op::AvgPool2d { x, k, stride, pad } => {
                let desc = MPSGraphPooling2DOpDescriptor::descriptorWithKernelWidth_kernelHeight_strideInX_strideInY_dilationRateInX_dilationRateInY_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout(
                    *k, *k, *stride, *stride, 1, 1, *pad, *pad, *pad, *pad,
                    MPSGraphPaddingStyle::Explicit,
                    MPSGraphTensorNamedDataLayout::NCHW,
                )
                .context("invalid pooling descriptor")?;
                if matches!(node.op, Op::MaxPool2d { .. }) {
                    graph.maxPooling2DWithSourceTensor_descriptor_name(&*get(x)?, &desc, None)
                } else {
                    // `nn.AvgPool2d` counts the zero padding (count_include_pad=True)
                    desc.setIncludeZeroPadToAverage(true);
                    graph.avgPooling2DWithSourceTensor_descriptor_name(&*get(x)?, &desc, None)
                }
            }
            Op::UpsampleNearest { x, h, w } => {
                let s = shape(x);
                let (n, c, ih, iw) = (s[0], s[1], s[2], s[3]);
                ensure!(
                    h % ih == 0 && w % iw == 0,
                    "nearest upsampling by a non-integer factor ({ih}x{iw} -> {h}x{w})"
                );
                let (fy, fx) = (h / ih, w / iw);
                let t = reshape(graph, &*get(x)?, &[n, c, ih, 1, iw, 1]);
                graph.broadcastTensor_toShape_name(&t, &shape_of(&[n, c, ih, fy, iw, fx]), None)
            }
            Op::Reshape(x, _) => get(x)?,
            Op::Permute(x, perm) => graph.transposeTensor_permutation_name(&*get(x)?, &axes(perm), None),
            Op::Concat(xs, axis) => {
                let parts = xs.iter().map(get).collect::<Result<Vec<_>>>()?;
                graph.concatTensors_dimension_name(&NSArray::from_retained_slice(&parts), *axis as isize, None)
            }
            Op::Slice { x, axis, start, end } => graph.sliceTensor_dimension_start_length_name(
                &*get(x)?,
                *axis,
                *start as isize,
                (*end - *start) as isize,
                None,
            ),
            Op::ReduceSum(x, ax) => graph.reductionSumWithTensor_axes_name(&*get(x)?, Some(&axes(ax)), None),
            Op::ReduceMean(x, ax) => graph.meanOfTensor_axes_name(&*get(x)?, &axes(ax), None),
            Op::BatchNorm {
                x,
                gamma,
                beta,
                eps,
                mode,
                relu,
            } => {
                let BnMode::Eval { mean, var } = mode else {
                    bail!("batch norm in training mode has no GPU replay");
                };
                let (gm, bt) = (&g.nodes[gamma.0].data, &g.nodes[beta.0].data);
                let scale: Vec<f32> = gm.iter().zip(var).map(|(g, v)| g / (v + eps).sqrt()).collect();
                let shift: Vec<f32> = bt.iter().zip(mean).zip(&scale).map(|((b, m), s)| b - m * s).collect();
                let rank = shape(x).len();
                let mut cshape = vec![1usize; rank];
                cshape[1] = scale.len();
                let y = graph.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &*get(x)?,
                    &constant(graph, &scale, &cshape),
                    None,
                );
                let y = graph.additionWithPrimaryTensor_secondaryTensor_name(&y, &constant(graph, &shift, &cshape), None);
                if *relu {
                    graph.reLUWithTensor_name(&y, None)
                } else {
                    y
                }
            }
            Op::L2Normalize(x, axis) => {
                let xt = get(x)?;
                let sq = graph.squareWithTensor_name(&xt, None);
                let sum = graph.reductionSumWithTensor_axes_name(&sq, Some(&axes(&[*axis])), None);
                let norm = graph.squareRootWithTensor_name(&sum, None);
                let norm = graph.maximumWithPrimaryTensor_secondaryTensor_name(&norm, &scalar(graph, 1e-12), None);
                graph.divisionWithPrimaryTensor_secondaryTensor_name(&xt, &norm, None)
            }
            Op::Dropout { x, .. } => get(x)?,
            other => bail!("{} has no GPU replay", op_name(other)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Model, ModelKind};
    use crate::tensor::Ctx;

    /// CPU forward of `model` on `x`: the tape, the input node and the outputs.
    fn cpu(model: &Model, x: &[f32], shape: &[usize]) -> (Graph, NodeId, Vec<(&'static str, NodeId)>) {
        let mut g = Graph::new();
        let input = g.from_data(x.to_vec(), shape);
        let out = model.forward_named(&mut Ctx::eval(), &mut g, &[input]).unwrap();
        (g, input, out)
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (s >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }

    /// The general detector and pose networks: a capture replayed on a new input matches the CPU
    /// forward pass on that input.
    #[test]
    fn general_networks_match_the_cpu() {
        if !crate::tensor::gpu_available() {
            eprintln!("SKIP: no Metal device");
            return;
        }
        let Some(root) = crate::data::repo_root() else { return };
        for (kind, shape) in [("detector", vec![1, 3, 256, 320]), ("pose", vec![1, 3, 192, 192])] {
            let path = root.join("weights").join(format!("{kind}_general.pt"));
            let Ok((cfg, k, w)) = crate::models::load(&path) else {
                eprintln!("SKIP: {} is unreadable (git lfs pull)", path.display());
                continue;
            };
            assert_eq!(k, kind.parse::<ModelKind>().unwrap());
            let model = Model::build(k, &cfg, &w).unwrap();
            let n: usize = shape.iter().product();
            let (g, input, out) = cpu(&model, &noise(n, 1), &shape);
            let ids: Vec<NodeId> = out.iter().map(|o| o.1).collect();
            let mut gpu = GpuForward::capture(&g, input, &ids).unwrap();
            let x2 = noise(n, 2);
            let got = gpu.run(&x2).unwrap();
            let (g2, _, out2) = cpu(&model, &x2, &shape);
            for ((name, id), values) in out2.iter().zip(&got) {
                let want = g2.value(*id);
                assert_eq!(values.len(), want.len(), "{kind}.{name}");
                let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
                let err = want.iter().zip(values).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(err <= 1e-3 * scale, "{kind}.{name}: max error {err} (scale {scale})");
            }
        }
    }
}
