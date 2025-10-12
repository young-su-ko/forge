import torch

from forge.inference.solvers import ODESolver


class FlowSimulator:
    def __init__(
        self, velocity_model, raygun, guidance_scale=0.0, t_steps=100, device="cuda"
    ):
        self.velocity_model = velocity_model.eval()
        self.guidance_scale = guidance_scale
        self.t_steps = t_steps
        self.device = device
        self.solver = ODESolver(self.velocity_model, self.guidance_scale)
        self.raygun = raygun.to(self.device)
        self.raygun_encoder = self.raygun.encoder

    @torch.no_grad()
    def sample(
        self, esm_condition: torch.Tensor | None, length: int, n_samples: int
    ) -> str:
        if esm_condition is not None:
            condition = self.raygun_encoder(esm_condition)
            condition = condition.repeat(n_samples, 1, 1)
        else:
            if self.guidance_scale > 0:
                raise ValueError("Cannot use guidance without a condition.")
            condition = torch.zeros(
                n_samples, 50, 1280, device=self.device
            )  # Fixed shape for Raygun embeddings

        x0 = torch.randn_like(condition)
        xt = self.solver.solve(x0, self.t_steps, condition)
        lengths = torch.full(
            (n_samples,), fill_value=length, device=self.device, dtype=torch.long
        )

        return self.raygun.get_sequences_from_fixed(xt, lengths)

class ValFlowSimulator:
    def __init__(
        self, velocity_model, guidance_scale=0.0, t_steps=100
    ):
        self.velocity_model = velocity_model.eval()
        self.guidance_scale = guidance_scale
        self.t_steps = t_steps
        self.solver = ODESolver(self.velocity_model, self.guidance_scale)

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor,
    ) -> torch.Tensor:
        x0 = torch.randn_like(condition)
        xt = self.solver.solve(x0, self.t_steps, condition)

        return xt