"""Native Torch counterpart of yam/stable.py; no gradients through the VLA.

Q10/min2, bounded tanh editor, chunk TD, Polyak target Q, entropy temperature.
The ResNet uses GroupNorm and independent current/next crops. Checkpoints are
not interchangeable with JAX; only the transport and episode schemas are shared.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Settings:
    candidates: int = 8
    edits: int = 8
    edit_scale: float = 0.05
    discount: float = 0.99
    tau: float = 0.005
    lr: float = 1e-4
    editor_lr: float = 3e-5
    temperature_lr: float = 1e-5
    init_temperature: float = 0.01
    initial_logstd: float = -3.0
    gradient_clip: float = 1.0
    filters: int = 64
    stages: tuple = (3, 4, 6, 3)
    hidden: tuple = (256, 256, 256)
    image_latent: int = 512
    state_latent: int = 64
    image_size: int = 224
    batch_size: int = 8
    updates: int = 40
    editor_interval: int = 20
    seed: int = 42
    actor_mode: str = "frozen"

    def __post_init__(self):
        if not 1 <= self.edits <= self.candidates or self.actor_mode != "frozen":
            raise ValueError("Requires frozen base and 1 <= edits <= candidates")
        if not 0 < self.edit_scale <= 0.2 or self.filters % 4 or self.image_size % 32:
            raise ValueError("Invalid edit bound/encoder geometry")
        if min(self.updates, self.batch_size, self.editor_interval) < 1:
            raise ValueError("Training counts must be positive")

    def dictionary(self):
        # JSON-normalized identity, including tuples.
        import json

        return json.loads(json.dumps(asdict(self)))


def gn(n):
    return nn.GroupNorm(4, n, eps=1e-5)


class Block(nn.Module):
    def __init__(self, inp, out, stride):
        super().__init__()
        self.net = nn.Sequential(
            gn(inp),
            nn.ReLU(),
            nn.Conv2d(inp, out, 3, stride, 1, bias=False),
            gn(out),
            nn.ReLU(),
            nn.Conv2d(out, out, 3, 1, 1, bias=False),
        )
        self.skip = (
            nn.Identity()
            if inp == out and stride == 1
            else nn.Conv2d(inp, out, 1, stride, bias=False)
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        layers = [nn.Conv2d(9, c.filters, 7, 2, 3, bias=False), nn.MaxPool2d(3, 2, 1)]
        n = c.filters
        for i, depth in enumerate(c.stages):
            for j in range(depth):
                out = c.filters * 2**i
                layers.append(Block(n, out, 2 if i and j == 0 else 1))
                n = out
        # Keep the spatial flattening of the repository's ResNetV2 encoder.
        side = c.image_size // (4 * 2 ** (len(c.stages) - 1))
        layers += [
            gn(n),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n * side * side, c.image_latent),
            nn.LayerNorm(c.image_latent),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mlp(inp, hidden, out):
    layers = []
    for width in hidden:
        layers += [nn.Linear(inp, width), nn.LayerNorm(width), nn.ReLU()]
        inp = width
    return nn.Sequential(*layers, nn.Linear(inp, out))


class Head(nn.Module):
    def __init__(self, c, out):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(14, c.state_latent), nn.LayerNorm(c.state_latent), nn.Tanh()
        )
        self.net = mlp(c.image_latent + c.state_latent + 420, c.hidden, out)

    def forward(self, z, s, a):
        return self.net(torch.cat([z, self.state(s), a], -1))


class Agent(nn.Module):
    def __init__(self, settings=None, device="cpu", training=True):
        super().__init__()
        self.cfg = c = settings or Settings()
        torch.manual_seed(c.seed)
        self.encoder = Encoder(c)
        self.critic = nn.ModuleList([Head(c, 1) for _ in range(10)])
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.editor = Head(c, 840)
        nn.init.zeros_(self.editor.net[-1].weight)
        nn.init.zeros_(self.editor.net[-1].bias)
        with torch.no_grad():
            self.editor.net[-1].bias[420:] = c.initial_logstd
        self.log_temperature = nn.Parameter(torch.tensor(math.log(c.init_temperature)))
        active = torch.ones(30, 14)
        active[:, [6, 13]] = 0
        self.register_buffer("active", active.flatten())
        self.to(device)
        self.generator = torch.Generator(device=device).manual_seed(c.seed)
        self.updates = 0
        if training:
            self.qopt = torch.optim.Adam(
                [*self.encoder.parameters(), *self.critic.parameters()], lr=c.lr
            )
            self.eopt = torch.optim.Adam(self.editor.parameters(), lr=c.editor_lr)
            self.topt = torch.optim.Adam([self.log_temperature], lr=c.temperature_lr)

    @property
    def device(self):
        return self.log_temperature.device

    def images(self, rgb, augment=False):
        x = torch.as_tensor(rgb, device=self.device).float()
        # B,3,H,W,3 -> separate RGB views for independent augmentation.
        _b, v, h, w, ch = x.shape
        if v != 3 or ch != 3:
            raise ValueError("Expected three RGB cameras")
        views = []
        for i in range(3):
            im = x[:, i].permute(0, 3, 1, 2)
            if augment:
                cropped = []
                hh, ww = round(h * 0.95), round(w * 0.95)
                for item in im:
                    y = int(
                        torch.randint(
                            h - hh + 1, (), generator=self.generator, device=self.device
                        )
                    )
                    xx = int(
                        torch.randint(
                            w - ww + 1, (), generator=self.generator, device=self.device
                        )
                    )
                    cropped.append(item[:, y : y + hh, xx : xx + ww])
                im = torch.stack(cropped)
            views.append(
                F.interpolate(
                    im,
                    size=(self.cfg.image_size,) * 2,
                    mode="bilinear",
                    align_corners=False,
                )
                / 127.5
                - 1
            )
        return torch.cat(views, 1)

    @staticmethod
    def qs(heads, z, s, a):
        return torch.stack([h(z, s, a).squeeze(-1) for h in heads])

    def edit(self, z, s, a):
        mean, logstd = self.editor(z, s, a).chunk(2, -1)
        logstd = logstd.clamp(-20, 2)
        noise = torch.randn(mean.shape, device=self.device, generator=self.generator)
        u = mean + logstd.exp() * noise
        lp = -0.5 * (noise.square() + 2 * logstd + math.log(2 * math.pi))
        lp -= 2 * (math.log(2) - u - F.softplus(-2 * u))
        return self.cfg.edit_scale * u.tanh() * self.active, (lp * self.active).sum(-1)

    @torch.no_grad()
    def select_encoded(self, z, s, base):
        b, n, d = base.shape
        e = self.cfg.edits
        if n < e or d != 420:
            raise ValueError("Candidate geometry mismatch")
        a = base[:, :e].reshape(-1, 420)
        delta, _ = self.edit(z.repeat_interleave(e, 0), s.repeat_interleave(e, 0), a)
        edited = (a + delta).clamp(-1, 1).reshape(b, e, 420)
        choices = torch.cat([base, edited], 1)
        k = choices.shape[1]
        q = self.qs(
            self.target,
            z.repeat_interleave(k, 0),
            s.repeat_interleave(k, 0),
            choices.reshape(-1, 420),
        )
        pair = torch.randperm(10, device=self.device, generator=self.generator)[:2]
        scores = q[pair].amin(0).reshape(b, k)
        if not torch.isfinite(q).all():
            raise FloatingPointError("Nonfinite Q")
        selected = scores.argmax(-1)
        return choices[torch.arange(b, device=self.device), selected], {
            "scores": scores,
            "selected": selected,
        }

    @torch.no_grad()
    def select(self, rgb, states, base):
        z = self.encoder(self.images(rgb))
        return self.select_encoded(
            z,
            torch.as_tensor(states, device=self.device, dtype=torch.float32),
            torch.as_tensor(base, device=self.device, dtype=torch.float32),
        )

    def _tensor(self, b, k):
        return torch.as_tensor(b[k], device=self.device, dtype=torch.float32)

    def critic_loss(self, b):
        c = self.cfg
        with torch.no_grad():
            zn = self.encoder(self.images(b["next_rgb"], True))
            ns = self._tensor(b, "next_states")
            nxt, _ = self.select_encoded(zn, ns, self._tensor(b, "next_candidates"))
            pair = torch.randperm(10, device=self.device, generator=self.generator)[:2]
            nq = self.qs(self.target, zn, ns, nxt)[pair].amin(0)
            target = (
                self._tensor(b, "rewards")
                + c.discount ** self._tensor(b, "steps") * self._tensor(b, "masks") * nq
            )
        z = self.encoder(self.images(b["rgb"], True))
        q = self.qs(
            self.critic, z, self._tensor(b, "states"), self._tensor(b, "actions")
        )
        loss = (q - target[None]).square().mean()
        return loss, {
            "critic_loss": loss.detach(),
            "target_mean": target.mean(),
            "q_mean": q.detach().mean(),
        }

    def step_optimizer(self, opt, params):
        params = list(params)
        norm = nn.utils.clip_grad_norm_(
            params, self.cfg.gradient_clip, error_if_nonfinite=True
        )
        opt.step()
        return float(norm)

    def critic_update(self, batches, terminal_batches=(), terminal_weight=0.25):
        self.qopt.zero_grad(set_to_none=True)
        metrics = {}
        for group, weight, prefix in (
            (batches, 1.0, ""),
            (terminal_batches, terminal_weight, "terminal_"),
        ):
            for b in group:
                if prefix and torch.any(self._tensor(b, "masks") != 0):
                    raise ValueError("Auxiliary terminal must not bootstrap")
                loss, m = self.critic_loss(b)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite critic loss")
                (loss * weight / len(group)).backward()
                for k, v in m.items():
                    key = prefix + k
                    metrics[key] = metrics.get(key, 0) + float(v) / len(group)
        if not batches:
            raise ValueError("Empty batch")
        metrics["critic_grad_norm"] = self.step_optimizer(
            self.qopt, [*self.encoder.parameters(), *self.critic.parameters()]
        )
        with torch.no_grad():
            for t, p in zip(
                self.target.parameters(), self.critic.parameters(), strict=True
            ):
                t.lerp_(p, self.cfg.tau)
        self.updates += 1
        return metrics

    def editor_update(self, batches):
        if not batches:
            raise ValueError("Empty batch")
        self.eopt.zero_grad(set_to_none=True)
        self.topt.zero_grad(set_to_none=True)
        self.critic.requires_grad_(False)
        metrics = {}
        try:
            for b in batches:
                with torch.no_grad():
                    z = self.encoder(self.images(b["rgb"], True))
                s, a = self._tensor(b, "states"), self._tensor(b, "actions")
                delta, lp = self.edit(z, s, a)
                q = self.qs(self.critic, z, s, (a + delta).clamp(-1, 1)).mean(0)
                loss = (self.log_temperature.exp().detach() * lp - q).mean()
                entropy = -lp.detach().mean()
                tloss = self.log_temperature.exp() * (entropy + self.active.sum() / 2)
                if not torch.isfinite(loss + tloss):
                    raise FloatingPointError("Nonfinite editor loss")
                ((loss + tloss) / len(batches)).backward()
                for k, v in {
                    "editor_loss": loss,
                    "entropy_unscaled": entropy,
                    "edit_abs_mean": delta.abs().mean(),
                }.items():
                    metrics[k] = metrics.get(k, 0) + float(v.detach()) / len(batches)
            metrics["editor_grad_norm"] = self.step_optimizer(
                self.eopt, self.editor.parameters()
            )
            self.step_optimizer(self.topt, [self.log_temperature])
        finally:
            self.critic.requires_grad_(True)
        metrics["temperature"] = float(self.log_temperature.exp().detach())
        return metrics

    def save(self, folder):
        # Safetensors is the serving payload. Optimizer/RNG state stays on learner.
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            str(folder / "policy.safetensors"),
        )
        torch.save(
            {
                "qopt": self.qopt.state_dict(),
                "eopt": self.eopt.state_dict(),
                "topt": self.topt.state_dict(),
                "rng": self.generator.get_state(),
                "updates": self.updates,
            },
            folder / "training.pt",
        )

    def restore(self, folder, training=False):
        self.load_state_dict(
            load_file(str(folder / "policy.safetensors"), device=str(self.device)),
            strict=True,
        )
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("Nonfinite checkpoint")
        if training:
            data = torch.load(
                folder / "training.pt", map_location=self.device, weights_only=True
            )
            for name in ("qopt", "eopt", "topt"):
                getattr(self, name).load_state_dict(data[name])
            self.generator.set_state(data["rng"].cpu())
            self.updates = data["updates"]
