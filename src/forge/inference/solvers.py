import torch


# Euler method only for now
class ODESolver:
    def __init__(self, velocity_model, guidance_scale: float = 0.0):
        self.velocity_model = velocity_model
        self.guidance_scale = guidance_scale

    def _velocity(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, length: torch.Tensor
    ) -> torch.Tensor:
        if self.guidance_scale == 0.0:
            return self.velocity_model(x, t, condition, length)
        v_cond = self.velocity_model(x, t, condition, length)
        v_uncond = self.velocity_model(x, t, torch.zeros_like(condition), length)
        return (1-self.guidance_scale) * v_uncond + self.guidance_scale * v_cond

    def solve(
        self, x0: torch.Tensor, t_steps: int, condition: torch.Tensor, length: torch.Tensor
    ) -> torch.Tensor:
        x = x0
        dt = 1.0 / t_steps
        for i in range(t_steps):
            t = torch.full((x.size(0),), i * dt, device=x.device)
            x = x + dt * self._velocity(x, t, condition, length)
        return x

    # def compute_likelihood(
    #     self,
    #     x_1: torch.Tensor,
    #     log_p0: Callable[[Tensor], Tensor],
    #     t_steps: int,
    #     condition: Tensor,
    #     return_intermediates: bool = False,
    # ) -> Union[Tuple[Tensor, Tensor], Tuple[list, Tensor]]:
    #     """
    #     Compute log likelihood by solving ODE backwards from t=1 to t=0.

    #     Args:
    #         x_1: Target sample at t=1 (e.g., data samples)
    #         log_p0: Log probability function of the source distribution at t=0
    #         t_steps: Number of integration steps
    #         condition: Conditioning information
    #         return_intermediates: Whether to return intermediate trajectories

    #     Returns:
    #         If return_intermediates=False: (x_0, log_likelihood)
    #         If return_intermediates=True: (trajectory_list, log_likelihood)
    #     """
    #     device = x_1.device
    #     batch_size = x_1.shape[0]

    #     # Initialize trajectory and log determinant
    #     x = x_1.clone()
    #     log_det = torch.zeros(batch_size, device=device)

    #     # Fixed random projection for Hutchinson estimator (same across time steps)
    #     if not exact_divergence:
    #         z = (torch.randn_like(x_1) < 0).float() * 2.0 - 1.0

    #     # Store intermediates if requested
    #     trajectory = [x.clone()] if return_intermediates else None

    #     # Time step (negative because we go backwards: t=1 -> t=0)
    #     dt = -1.0 / t_steps

    #     for i in range(t_steps):
    #         # Current time (going from 1.0 to 0.0)
    #         t_val = 1.0 - i / t_steps
    #         t = torch.full((batch_size,), t_val, device=device)

    #         # Enable gradients for divergence computation
    #         x.requires_grad_(True)

    #         # Compute velocity
    #         ut = self._velocity(x, t, condition)

    #         # Hutchinson estimator: E[z^T ∇_x(u^T z)]
    #         ut_dot_z = torch.einsum(
    #             "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
    #         )

    #         grad_ut_dot_z = torch.autograd.grad(
    #             ut_dot_z.sum(), x, create_graph=False
    #         )[0]

    #         div = torch.einsum(
    #             "ij,ij->i",
    #             grad_ut_dot_z.flatten(start_dim=1),
    #             z.flatten(start_dim=1),
    #         )

    #         # Euler step (backward in time)
    #         with torch.no_grad():
    #             x = x + dt * ut
    #             log_det = log_det + dt * div

    #             if return_intermediates:
    #                 trajectory.append(x.clone())

    #     # Final sample at t=0
    #     x_0 = x

    #     # Compute log probability under source distribution
    #     source_log_p = log_p0(x_0)

    #     # Total log likelihood (note: positive log_det because we integrated backwards)
    #     log_likelihood = source_log_p + log_det

    #     if return_intermediates:
    #         return trajectory, log_likelihood
    #     else:
    #         return x_0, log_likelihood
