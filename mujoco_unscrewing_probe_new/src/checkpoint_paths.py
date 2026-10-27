from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def resolve_checkpoint_path(value: str, model_root: Path) -> Path:
    """Rebase an archived checkpoint path only inside the same named bundle.

    Keep the immutable contract unchanged. Never substitute a different seed
    or checkpoint basename when a copied repository has a different root.
    """
    recorded = Path(value)
    native = recorded if recorded.is_absolute() else ROOT / recorded
    if native.is_file():
        return native
    parts = PurePosixPath(value.replace("\\", "/")).parts
    marker = ("outputs", "current", "models", model_root.name)
    for index in range(len(parts) - len(marker)):
        if parts[index:index + len(marker)] == marker:
            suffix = parts[index + len(marker):]
            if suffix and suffix[0] == "seeds" and ".." not in suffix:
                relocated = model_root.joinpath(*suffix)
                if relocated.is_file():
                    return relocated
    raise FileNotFoundError(f"Checkpoint absent at recorded or relocated bundle path: {value}")

