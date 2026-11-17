"""Train arena detectors from the hand-labelled frames of scripts/label_arena.py.

One network per arena definition (`configs/arenas/<type>.yaml`): the animalpose pose network
(ResNet-18 + FPN, ImageNet start) over the whole top-view frame at 384x288, one heatmap and offset
per arena point, trained with the pose loss. It is saved as an `arena` checkpoint whose
`model_cfg["arena"]` carries the definition (point names, floor, zones, input size), so the apps
build the layout from the predicted points without knowing the arena type. Weights are stored in
half precision (inference is unaffected; the file is half the size).

Augmentation: rotations up to 12°, scale 0.8-1.2, shifts up to 10%, horizontal and vertical flips
(the points are put back into the definition's order afterwards; transforms that push a point out
of the frame are redrawn), brightness, contrast, gamma, blur, noise, and dark blobs for a mouse or
a hand.

Runs per type and what they measure:
  heldout    a fifth of each experiment's videos left out: error on new recordings of a known setup
  loeo:<e>   experiment <e> left out (types with several setups): error on a setup never seen
  final      every labelled frame: the network that ships

    python scripts/train_arena.py --labels ~/upload/arena_labels --out ~/upload/arena [--type all]

Writes <out>/runs/arena_<type>.pt, <out>/report_<type>.json and <out>/report.txt. Frames are read
from the videos once and cached in <out>/frames/<type>/.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from animalpose import arena as arenas  # noqa: E402
from animalpose.pose.model import PoseNet, build_targets, decode, pose_loss  # noqa: E402

IN_W, IN_H = 384, 288
STRIDE = 4
BACKBONE = {"name": "resnet18", "out_channels": 64, "pretrained": True}
LARCC_FOLDERS = Path.home() / "upload/pose_3d"  # paths3d.src(experiment) -> the videos


def video_path(defn, exp, stem):
    if defn.sources.get("folder"):
        return Path(defn.sources["folder"]) / f"{stem}.mp4"
    sys.path.insert(0, str(LARCC_FOLDERS))
    from paths3d import src

    return src(exp) / f"{stem}.mp4"


def load_samples(defn, labels_dir, cache):
    """Every labelled frame (grey image cached as PNG) with its points in saved order."""
    data = json.loads((labels_dir / f"{defn.name}.json").read_text())
    cache.mkdir(parents=True, exist_ok=True)
    samples = []
    for key, video in sorted(data["videos"].items()):
        exp, stem = key.split("/", 1)
        missing = [f for f in video["frames"] if not (cache / f"{exp}__{stem}__{f}.png").exists()]
        if missing:
            cap = cv2.VideoCapture(str(video_path(defn, exp, stem)))
            for f in sorted(missing, key=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, frame = cap.read()
                if ok:
                    cv2.imwrite(str(cache / f"{exp}__{stem}__{f}.png"), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for f, pts in video["frames"].items():
            path = cache / f"{exp}__{stem}__{f}.png"
            if path.exists():
                ordered = np.asarray(defn.ordered(pts), np.float32)
                samples.append({"exp": exp, "video": key, "frame": f, "path": path, "pts": ordered})
    return samples


# ---------------------------------------------------------------- data


class Frames(torch.utils.data.Dataset):
    """Whole frames scaled to IN_W x IN_H, as the RGB [0, 1] tensor the pose network takes; with
    `train`, augmented and drawn so every experiment is seen equally often."""

    def __init__(self, samples, defn, train, length=None):
        self.samples, self.defn, self.train = samples, defn, train
        self.length = length or len(samples)
        self.images = {s["path"]: cv2.imread(str(s["path"]), cv2.IMREAD_GRAYSCALE) for s in samples}
        self.by_exp = {}
        for s in samples:
            self.by_exp.setdefault(s["exp"], []).append(s)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        s = random.choice(self.by_exp[random.choice(list(self.by_exp))]) if self.train else self.samples[i]
        img = self.images[s["path"]].astype(np.float32)
        h, w = img.shape
        m = np.array([[IN_W / w, 0, 0], [0, IN_H / h, 0], [0, 0, 1]], np.float64)
        if self.train:
            m = self.geometric(s["pts"], m)
        img = cv2.warpAffine(img, m[:2], (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        pts = (np.c_[s["pts"], np.ones(len(s["pts"]))] @ m[:2].T).astype(np.float32)
        if self.train:
            pts = np.asarray(self.defn.ordered(pts.tolist()), np.float32)
            img = photometric(img)
        x = torch.from_numpy(np.repeat((img / 255.0)[None], 3, 0).astype(np.float32))
        return x, torch.from_numpy(pts), torch.from_numpy(m.astype(np.float32))

    @staticmethod
    def geometric(pts, base):
        """A random affine (on top of `base`) that keeps every point inside the frame."""
        for _ in range(20):
            a = cv2.getRotationMatrix2D((IN_W / 2, IN_H / 2), random.uniform(-12, 12), random.uniform(0.8, 1.2))
            a = np.vstack([a, [0, 0, 1]])
            a[:2, 2] += (random.uniform(-0.1, 0.1) * IN_W, random.uniform(-0.1, 0.1) * IN_H)
            if random.random() < 0.5:
                a = np.array([[-1, 0, IN_W], [0, 1, 0], [0, 0, 1]], np.float64) @ a
            if random.random() < 0.5:
                a = np.array([[1, 0, 0], [0, -1, IN_H], [0, 0, 1]], np.float64) @ a
            m = a @ base
            p = np.c_[pts, np.ones(len(pts))] @ m[:2].T
            if (p >= 2).all() and (p[:, 0] < IN_W - 2).all() and (p[:, 1] < IN_H - 2).all():
                return m
        return base


def photometric(img):
    img = img * random.uniform(0.6, 1.4) + random.uniform(-40, 40)
    img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** random.uniform(0.7, 1.4)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 2.0))
    for _ in range(random.randint(0, 3)):  # a mouse, a hand, a cable
        centre = (int(random.uniform(0, IN_W)), int(random.uniform(0, IN_H)))
        axes = (int(random.uniform(5, 30)), int(random.uniform(3, 15)))
        cv2.ellipse(img, centre, axes, random.uniform(0, 180), 0, 360, random.uniform(0, 80), -1)
    img = img + np.random.normal(0, random.uniform(0, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


# ---------------------------------------------------------------- train / evaluate


def train(defn, samples, steps, device, log):
    net = PoseNet(len(defn.points), BACKBONE).to(device)
    loader = torch.utils.data.DataLoader(
        Frames(samples, defn, train=True, length=steps * 16), batch_size=16, num_workers=8, persistent_workers=True
    )
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4, total_steps=steps, pct_start=0.1)
    out_hw = (IN_H // STRIDE, IN_W // STRIDE)
    net.train()
    t0 = time.time()
    for step, (x, pts, _) in enumerate(loader):
        state = torch.full(pts.shape[:2], 2)
        tgt = build_targets(pts, state, out_hw, STRIDE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = net(x.to(device, non_blocking=True))
        loss = pose_loss({k: v.float() for k, v in out.items()}, tgt)["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"    step {step:5d}  loss {loss.item():.4f}  {time.time() - t0:.0f}s")
    return net.eval()


@torch.no_grad()
def evaluate(net, defn, samples, device):
    """Error per point in original frame pixels, decoded as the apps decode (peak + offset)."""
    data = Frames(samples, defn, train=False)
    errs, per_frame = [], []
    for i in range(len(data)):
        x, _, m = data[i]
        out = net(x[None].to(device))
        pred, conf = decode(out["heat"].float(), out["offset"].float(), STRIDE)
        pred = pred[0].cpu().numpy()
        back = np.linalg.inv(m.numpy().astype(np.float64))
        pred = np.c_[pred, np.ones(len(pred))] @ back[:2].T
        e = np.linalg.norm(pred - samples[i]["pts"], axis=1)
        errs.append(e)
        per_frame.append(
            {
                "video": samples[i]["video"],
                "frame": samples[i]["frame"],
                "max_px": float(e.max()),
                "min_conf": float(conf.min()),
            }
        )
    e = np.concatenate(errs)
    return {
        "frames": len(samples),
        "median_px": float(np.median(e)),
        "p90_px": float(np.percentile(e, 90)),
        "max_px": float(e.max()),
        "within_5px": float((e <= 5).mean()),
        "worst": sorted(per_frame, key=lambda f: -f["max_px"])[:5],
    }


def save(net, defn, path, frames):
    model_cfg = {
        "backbone": {**BACKBONE, "pretrained": False},
        "num_keypoints": len(defn.points),
        "embed_dim": 64,
        "arena": defn.as_model_cfg((IN_W, IN_H)),
    }
    state = {k: (v.half() if v.is_floating_point() else v).detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"kind": "arena", "model_cfg": model_cfg, "state_dict": state, "trained_on_frames": frames}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=Path.home() / "upload/arena_labels")
    parser.add_argument("--out", type=Path, default=Path.home() / "upload/arena")
    parser.add_argument("--definitions", type=Path, default=arenas.DEFINITIONS)
    parser.add_argument("--type", default="all", help="an arena type, or all")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (args.out / "runs").mkdir(parents=True, exist_ok=True)
    names = arenas.available(args.definitions) if args.type == "all" else [args.type]
    rows = []

    def log(text):
        print(text, flush=True)

    for name in names:
        if not (args.labels / f"{name}.json").exists():
            log(f"{name}: no labels")
            continue
        defn = arenas.load(name, args.definitions)
        samples = load_samples(defn, args.labels, args.out / "frames" / name)
        exps = sorted({s["exp"] for s in samples})
        log(f"== {name}: {len(samples)} frames, {len({s['video'] for s in samples})} videos, experiments {exps}")
        held = set()
        for e in exps:
            vids = sorted({s["video"] for s in samples if s["exp"] == e})
            held |= set(vids[2::5] or vids[:1])
        runs = [("heldout", [s for s in samples if s["video"] not in held], [s for s in samples if s["video"] in held])]
        if len(exps) > 1:
            for e in exps:
                runs.append((f"loeo:{e}", [s for s in samples if s["exp"] != e], [s for s in samples if s["exp"] == e]))
        runs.append(("final", samples, []))
        report = {"type": name, "points": defn.points, "frames": len(samples), "runs": {}}
        for run, train_set, test_set in runs:
            log(f"  -- {run}: train {len(train_set)} frames, test {len(test_set)}")
            net = train(defn, train_set, args.steps, device, log)
            if test_set:
                stats = evaluate(net, defn, test_set, device)
                report["runs"][run] = stats
                rows.append((name, run, stats))
                log(
                    f"     {run}: median {stats['median_px']:.1f} px, 90% {stats['p90_px']:.1f} px, "
                    f"max {stats['max_px']:.1f} px, {100 * stats['within_5px']:.0f}% within 5 px"
                )
            else:
                path = args.out / "runs" / f"arena_{name}.pt"
                save(net, defn, path, len(samples))
                log(f"     saved {path}")
        (args.out / f"report_{name}.json").write_text(json.dumps(report, indent=2))

    lines = [f"{'type':14s} {'test':26s} {'frames':>6s} {'median':>7s} {'90%':>6s} {'max':>6s} {'<=5px':>6s}"]
    for name, run, st in rows:
        lines.append(
            f"{name:14s} {run:26s} {st['frames']:6d} {st['median_px']:6.1f}  {st['p90_px']:5.1f}  "
            f"{st['max_px']:5.1f}  {100 * st['within_5px']:5.0f}%"
        )
    text = "Arena detector (pose network), error per point in frame pixels\n" + "\n".join(lines) + "\n"
    (args.out / "report.txt").write_text(text)
    log(text)


if __name__ == "__main__":
    main()
