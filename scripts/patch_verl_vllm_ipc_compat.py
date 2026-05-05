from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def patch_bucketed_weight_transfer() -> bool:
    path = ROOT / "third_party/verl/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
    text = path.read_text()

    old = """def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:\n    func, args = handle\n    list_args = list(args)\n    if device_id is not None:\n        list_args[6] = device_id\n    buffer = func(*list_args)\n    return buffer\n"""
    new = """def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:\n    func, args = handle\n    list_args = list(args)\n    if device_id is not None and len(list_args) > 6:\n        # Older/newer torch builds may serialize CUDA IPC handles with different\n        # tuple layouts. Only rewrite the device slot when it actually exists.\n        # On single-GPU setups the original handle is usually already correct.\n        list_args[6] = device_id\n    buffer = func(*list_args)\n    return buffer\n"""

    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Unexpected file contents in {path}")

    path.write_text(text.replace(old, new))
    return True


def patch_vllm_rollout() -> bool:
    path = ROOT / "third_party/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py"
    text = path.read_text()

    old = """        self.use_shm = not is_support_ipc()\n        if self.use_shm:\n"""
    new = """        force_shm = os.getenv(\"VERL_VLLM_FORCE_SHM\", \"\").strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}\n        self.use_shm = force_shm or (not is_support_ipc())\n        if self.use_shm:\n"""

    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Unexpected file contents in {path}")

    path.write_text(text.replace(old, new))
    return True


def main() -> int:
    changed = []
    if patch_bucketed_weight_transfer():
        changed.append("bucketed_weight_transfer.py")
    if patch_vllm_rollout():
        changed.append("vllm_rollout.py")

    if changed:
        print("Patched verl vLLM IPC compatibility:", ", ".join(changed))
    else:
        print("verl vLLM IPC compatibility patch already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
