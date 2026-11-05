
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch.func import jvp

from .local_dit import VoxCPMLocDiT


class CfmConfig(BaseModel):
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 1.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0


class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.reg_loss_type = cfm_params.reg_loss_type
        self.ratio_r_neq_t_range = cfm_params.ratio_r_neq_t_range
        self.noise_cond_prob_range = cfm_params.noise_cond_prob_range
        self.noise_cond_scale = cfm_params.noise_cond_scale

        self.in_channels = in_channels
        self.mean_mode = mean_mode

        self.estimator = estimator

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        b, _ = mu.shape
        t = patch_size
        z = torch.randn((b, self.in_channels, t), device=mu.device, dtype=mu.dtype) * temperature

        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        return self.solve_euler(
            x=z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    def optimized_scale(self, positive_flat: torch.Tensor, negative_flat: torch.Tensor):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        st_star = dot_product / squared_norm
        return st_star

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        t, _, dt = t_span[0], t_span[-1], t_span[0] - t_span[1]

        sol = []
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                # Classifier-Free Guidance inference introduced in VoiceBox
                b = x.size(0)
                x_in = torch.zeros([2 * b, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                mu_in = torch.zeros([2 * b, mu.size(1)], device=x.device, dtype=x.dtype)
                t_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                dt_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                cond_in = torch.zeros([2 * b, self.in_channels, cond.size(2)], device=x.device, dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                # not used now
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                cond_in[:b], cond_in[b:] = cond, cond

                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)

                if use_cfg_zero_star:
                    positive_flat = dphi_dt.view(b, -1)
                    negative_flat = cfg_dphi_dt.view(b, -1)
                    st_star = self.optimized_scale(positive_flat, negative_flat)
                    st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))
                else:
                    st_star = 1.0

                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - dt * dphi_dt
            t = t - dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return sol[-1]

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    def adaptive_loss_weighting(
        self, losses: torch.Tensor, mask: torch.Tensor | None = None, p: float = 0.0, epsilon: float = 1e-3
    ):
        weights = 1.0 / ((losses + epsilon).pow(p))
        if mask is not None:
            weights = weights * mask
        return weights.detach()

    def sample_r_t(self, x: torch.Tensor, mu: float = -0.4, sigma: float = 1.0, ratio_r_neq_t: float = 0.0):
        batch_size = x.shape[0]
        if self.t_scheduler == "log-norm":
            s_r = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu
            s_t = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu
            r = torch.sigmoid(s_r)
            t = torch.sigmoid(s_t)
        elif self.t_scheduler == "uniform":
            r = torch.rand(batch_size, device=x.device, dtype=x.dtype)
            t = torch.rand(batch_size, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported t_scheduler: {self.t_scheduler}")

        mask = torch.rand(batch_size, device=x.device, dtype=x.dtype) < ratio_r_neq_t
        r, t = torch.where(
            mask,
            torch.stack([torch.min(r, t), torch.max(r, t)], dim=0),
            torch.stack([t, t], dim=0),
        )

        return r.squeeze(), t.squeeze()

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        progress: float = 0.0,
    ):
        b, _, _ = x1.shape

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1)

        if cond is None:
            cond = torch.zeros_like(x1)

        noisy_mask = torch.rand(b, device=x1.device) > (
            1.0
            - (
                self.noise_cond_prob_range[0]
                + progress * (self.noise_cond_prob_range[1] - self.noise_cond_prob_range[0])
            )
        )
        cond = cond + noisy_mask.view(-1, 1, 1) * torch.randn_like(cond) * self.noise_cond_scale

        ratio_r_neq_t = (
            self.ratio_r_neq_t_range[0] + progress * (self.ratio_r_neq_t_range[1] - self.ratio_r_neq_t_range[0])
            if self.mean_mode
            else 0.0
        )

        r, t = self.sample_r_t(x1, ratio_r_neq_t=ratio_r_neq_t)
        r_ = r.detach().clone()
        t_ = t.detach().clone()
        z = torch.randn_like(x1)
        y = (1 - t_.view(-1, 1, 1)) * x1 + t_.view(-1, 1, 1) * z
        v = z - x1

        def model_fn(z_sample, r_sample, t_sample):
            return self.estimator(z_sample, mu, t_sample, cond, dt=t_sample - r_sample)

        if self.mean_mode:
            v_r = torch.zeros_like(r)
            v_t = torch.ones_like(t)
            from torch.backends.cuda import sdp_kernel

            with sdp_kernel(enable_flash=False, enable_mem_efficient=False):
                u_pred, dudt = jvp(model_fn, (y, r, t), (v, v_r, v_t))
            u_tgt = v - (t_ - r_).view(-1, 1, 1) * dudt
        else:
            u_pred = model_fn(y, r, t)
            u_tgt = v

        losses = F.mse_loss(u_pred, u_tgt.detach(), reduction="none").mean(dim=1)
        if tgt_mask is not None:
            weights = self.adaptive_loss_weighting(losses, tgt_mask.squeeze(1))
            loss = (weights * losses).sum() / torch.clamp(torch.sum(tgt_mask), min=1.0)
        else:
            loss = losses.mean()

        return loss
