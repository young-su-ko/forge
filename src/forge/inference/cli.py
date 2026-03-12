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
    n_samples: int = typer.Option(
        1, "--n-samples", help="Number of samples to generate."
    ),
    guidance_scale: float = typer.Option(
        3.0,
        "--guidance-scale",
        help="Classifier-free guidance scale (ignored for unconditional generation).",
    ),
    t_steps: int = typer.Option(100, "--t-steps", help="Number of solver timesteps."),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="Device to run on.",
    ),
):
    """Forge a binder (conditional if --target-sequence is provided)."""
    repo_or_dir = model if model else hf_repo
    use_conditioning = target_sequence is not None
    wrapper = InferenceWrapper.from_pretrained(
        repo_or_dir=repo_or_dir,
        guidance_scale=guidance_scale if use_conditioning else 0.0,
        t_steps=t_steps,
        device=device,
        map_location=device,
    )

    if target_sequence:
        print(wrapper.generate_binder(target_sequence, output_length, n_samples))
    else:
        print(wrapper.generate_unconditionally(output_length, n_samples))


def main():
    app()


if __name__ == "__main__":
    main()
