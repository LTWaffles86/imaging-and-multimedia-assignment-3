from pathlib import Path

output_dir = Path("images/output/")
if not output_dir.exists():
    output_dir.mkdir()


def output_path(input_path: Path, suffix: str) -> Path:
    return output_dir.joinpath(input_path.name).with_stem(
        input_path.stem + "_" + suffix
    )
