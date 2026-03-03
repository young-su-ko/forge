import torch
import typer
from forge.inference.wrapper import InferenceWrapper


app = typer.Typer(help="Forge inference CLI.")


@app.command()
def smith(
    model: str | None = typer.Option(
        None,
        "--model",
        help="Path to local model directory. If both --model and --hf-repo are provided, --model takes precedence.",
    ),
    hf_repo: str = typer.Option(
        "yk0/forge-e80",
        "--hf-repo",
        help="HuggingFace repository ID (e.g., 'yk0/forge-X') to load model from. Used as default if --model is not provided.",
    ),
    output_length: int = typer.Option(
        250, "--output-length", help="Length of generated sequence."
    ),
    target_sequence: str | None = typer.Option(
        None,
        "--target-sequence",
        help="Optional target protein sequence. If provided, generates a binder to this target.",
    ),
    target_sequence_b: str | None = typer.Option(
        None,
        "--target-sequence-b",
        help="Optional second target protein sequence for dual conditioning.",
    ),
    n_samples: int = typer.Option(
        1, "--n-samples", help="Number of samples to generate."
    ),
    guidance_scale: float = typer.Option(
        3.0,
        "--guidance-scale",
        help="Classifier-free guidance scale (ignored for unconditional generation).",
    ),
    guidance_scale_a: float | None = typer.Option(
        None,
        "--guidance-scale-a",
        help="Optional CFG scale for condition A (defaults to --guidance-scale).",
    ),
    guidance_scale_b: float | None = typer.Option(
        None,
        "--guidance-scale-b",
        help="Optional CFG scale for condition B (defaults to --guidance-scale).",
    ),
    dual_conditioning: str | None = typer.Option(
        None,
        "--dual-conditioning",
        help="Dual conditioning mode when both target sequences are provided.",
        metavar="[alternate|combined]",
    ),
    t_steps: int = typer.Option(100, "--t-steps", help="Number of solver timesteps."),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="Device to run on.",
    ),
):
    """Forge a binder (conditional if --target-sequence is provided)."""
    if dual_conditioning is not None and dual_conditioning.lower() not in {
        "alternate",
        "combined",
    }:
        raise typer.BadParameter(
            "--dual-conditioning must be one of: alternate, combined."
        )

    # Prefer local model if provided, otherwise use HF repo (which has a default)
    repo_or_dir = model if model else hf_repo
    scale_a = guidance_scale if guidance_scale_a is None else guidance_scale_a
    scale_b = guidance_scale if guidance_scale_b is None else guidance_scale_b

    use_conditioning = target_sequence is not None
    wrapper = InferenceWrapper.from_pretrained(
        repo_or_dir=repo_or_dir,
        guidance_scale=scale_a if use_conditioning else 0.0,
        t_steps=t_steps,
        device=device,
        map_location=device,
    )

    if dual_conditioning and not (target_sequence and target_sequence_b):
        raise typer.BadParameter(
            "--dual-conditioning requires both --target-sequence and --target-sequence-b."
        )

    if dual_conditioning:
        typer.echo(
            wrapper.generate_binder_dual(
                target_sequence_a=target_sequence,
                target_sequence_b=target_sequence_b,
                binder_length=output_length,
                n_samples=n_samples,
                dual_conditioning=dual_conditioning.lower(),
                guidance_scale_a=scale_a,
                guidance_scale_b=scale_b,
            )
        )
    elif target_sequence:
        typer.echo(
            wrapper.generate_binder(
                target_sequence,
                output_length,
                n_samples,
            )
        )
    else:
        typer.echo(wrapper.generate_unconditionally(output_length, n_samples))


@app.command()
def refine(
    model: str | None = typer.Option(
        None,
        "--model",
        help="Path to local model directory. If both --model and --hf-repo are provided, --model takes precedence.",
    ),
    hf_repo: str = typer.Option(
        "yk0/forge-e80",
        "--hf-repo",
        help="HuggingFace repository ID (e.g., 'yk0/forge-X') to load model from. Used as default if --model is not provided.",
    ),
    target_sequence: str = typer.Option(
        ...,
        "--target-sequence",
        help="Target protein sequence to condition on.",
    ),
    binder_sequence: str = typer.Option(
        ...,
        "--binder-sequence",
        help="Existing binder sequence used as the clean latent source.",
    ),
    output_length: int = typer.Option(
        250, "--output-length", help="Length of generated sequence."
    ),
    partial: float = typer.Option(
        0.5, "--partial", help="Partial diffusion fraction in [0,1]."
    ),
    n_samples: int = typer.Option(
        1, "--n-samples", help="Number of samples to generate."
    ),
    guidance_scale: float = typer.Option(
        3.0, "--guidance-scale", help="Classifier-free guidance scale."
    ),
    t_steps: int = typer.Option(100, "--t-steps", help="Number of solver timesteps."),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="Device to run on.",
    ),
):
    """Refine an existing binder via partial diffusion."""
    if not (0.0 <= partial <= 1.0):
        raise typer.BadParameter("--partial must be in [0, 1].")

    repo_or_dir = model if model else hf_repo
    wrapper = InferenceWrapper.from_pretrained(
        repo_or_dir=repo_or_dir,
        guidance_scale=guidance_scale,
        t_steps=t_steps,
        device=device,
        map_location=device,
    )

    typer.echo(
        wrapper.generate_binder(
            target_sequence=target_sequence,
            binder_length=output_length,
            n_samples=n_samples,
            partial=partial,
            binder_sequence=binder_sequence,
        )
    )

def main():
    app()


if __name__ == "__main__":
    main()
