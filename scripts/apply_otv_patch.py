"""Apply or verify the OTV changes in the pinned editable LLaMA-Factory checkout."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse


PINNED_COMMIT = "95ac3f2373b82662c1bd855c284d3379e6a763d3"
PATCH_FILE = Path(__file__).resolve().parents[1] / "patches" / "llamafactory-otv.patch"
PATCH_SHA256 = "7ba06d13ae7a6cd8a9df0a569b5442710b1dccbefedfc0ff1bccc0a8afbdba84"
EXPECTED_FILES = {
    "src/llamafactory/model/adapter.py": {
        "before": "db1ec6792e9b1b437071087257e973d092a25b4fb519eba58662b826ea0e9bf4",
        "after": "fd640514e095eaadaa4847bd880b969385b00c611642bff26d38ef4633af78b7",
    },
    "src/llamafactory/train/sft/trainer.py": {
        "before": "66e2c3dbbe0acda54090edb6e29a9e3fbc0bd7ed7aaf808bacac61d624e3172b",
        "after": "31e5fc95f9e7e2b95a4989b676f79c4ccc15fb0c10e297ab14eca081fc30a57a",
    },
}


def run_git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git command failed.")
    return result.stdout.strip()


def editable_checkout() -> Path:
    try:
        distribution = importlib.metadata.distribution("llamafactory")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Install requirements.txt in the active environment first.") from error

    metadata = json.loads(distribution.read_text("direct_url.json") or "{}")
    location = urlparse(metadata.get("url", ""))
    if (
        not metadata.get("dir_info", {}).get("editable")
        or location.scheme != "file"
        or location.netloc not in {"", "localhost"}
    ):
        raise RuntimeError("LLaMA-Factory must be installed as an editable Git dependency.")
    return Path(unquote(location.path)).resolve()


def apply_patch(source: Path, check_only: bool = False) -> str:
    source = source.resolve()
    if not source.is_dir():
        raise RuntimeError(f"Checkout directory does not exist: {source}")
    if run_git(source, "rev-parse", "--show-toplevel") != str(source):
        raise RuntimeError("The source directory must be the root of a dedicated Git checkout.")
    if run_git(source, "rev-parse", "HEAD") != PINNED_COMMIT:
        raise RuntimeError(f"Expected LLaMA-Factory commit {PINNED_COMMIT}; no files were changed.")
    if run_git(source, "diff", "--cached", "--name-only"):
        raise RuntimeError("The dependency has staged changes; no files were changed.")
    if hashlib.sha256(PATCH_FILE.read_bytes()).hexdigest() != PATCH_SHA256:
        raise RuntimeError("The OTV patch differs from its verified version; no files were changed.")

    original = {}
    states = []
    for relative, hashes in EXPECTED_FILES.items():
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Expected a regular source file: {relative}")
        original[relative] = path.read_bytes()
        digest = hashlib.sha256(original[relative]).hexdigest()
        state = next((name for name, expected in hashes.items() if digest == expected), None)
        if state is None:
            raise RuntimeError(f"Local edits detected in {relative}; no files were changed.")
        states.append(state)

    if all(state == "after" for state in states):
        return "OTV patch is already applied and verified."
    if check_only:
        raise RuntimeError("The OTV patch is not fully applied. Run scripts/apply_otv_patch.py first.")
    if not all(state == "before" for state in states):
        raise RuntimeError("The OTV patch is partially applied; no files were changed.")

    run_git(source, "apply", "--check", str(PATCH_FILE))
    try:
        run_git(source, "apply", str(PATCH_FILE))
        for relative, hashes in EXPECTED_FILES.items():
            digest = hashlib.sha256((source / relative).read_bytes()).hexdigest()
            if digest != hashes["after"]:
                raise RuntimeError(f"Patched file verification failed: {relative}")
    except Exception:
        for relative, content in original.items():
            (source / relative).write_bytes(content)
        raise
    return "OTV patch applied and verified."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, help="Dedicated checkout to patch; defaults to the editable installation."
    )
    parser.add_argument("--check", action="store_true", help="Verify the patch without changing any files.")
    args = parser.parse_args()
    try:
        source = args.source_dir if args.source_dir is not None else editable_checkout()
        print(apply_patch(source, check_only=args.check))
    except (RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
