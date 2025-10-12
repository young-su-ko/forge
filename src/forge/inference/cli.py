import click
import torch
from forge.inference.wrapper import InferenceWrapper
from forge.inference.hf_models import ForgeModelParam


@click.group()
def main():
    """Forge inference CLI."""
    pass

@main.command()
@click.option(
    "--model",
    type=ForgeModelParam(),
    default="forge-v0",
    show_default=True,
    help="Model to use.",
)
@click.option(
    "--output-length",
    default=250,
    type=int,
    show_default=True,
    help="Length of generated sequence.",
)
@click.option(
    "--target-sequence",
    default=None,
    type=str,
    help="Optional target protein sequence. If provided, generates a binder to this target.",
)
@click.option(
    "--n-samples",
    default=1,
    type=int,
    help="Number of samples to generate.",
)
@click.option(
    "--guidance-scale",
    default=1.0,
    type=float,
    show_default=True,
    help="Classifier-free guidance scale (ignored for unconditional generation).",
)
@click.option(
    "--t-steps",
    default=100,
    type=int,
    show_default=True,
    help="Number of solver timesteps.",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Device to run on.",
)
def smith(
    model, output_length, target_sequence, n_samples, guidance_scale, t_steps, device
):
    """Forge a binder (conditional if --target-sequence is provided)."""
    wrapper = InferenceWrapper.from_pretrained(
        repo_or_dir=model.repo_id,
        guidance_scale=guidance_scale if target_sequence else 0.0,
        t_steps=t_steps,
        device=device,
        map_location=device,
    )
    if target_sequence:
        click.echo(wrapper.generate_binder(target_sequence, output_length, n_samples))
    else:
        click.echo(wrapper.generate_unconditionally(output_length, n_samples))


if __name__ == "__main__":
    main()
