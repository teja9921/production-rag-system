import subprocess
import sys


STEPS = [
    ("retrieval_eval", [sys.executable, "-m", "evaluation.myset.scripts.run_eval_ranked"]),
    ("compute_metrics", [sys.executable, "-m", "evaluation.myset.scripts.compute_metrics"]),
    ("extract_failures", [sys.executable, "-m", "evaluation.myset.extract_failures"]),
    ("build_failure_index", [sys.executable, "-m", "evaluation.myset.build_failure_index"]),
]


def run_step(name: str, cmd: list[str]) -> None:
    print(f"[START] {name}: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed: {name} (exit_code={completed.returncode})")
    print(f"[DONE] {name}")


def main() -> None:
    for name, cmd in STEPS:
        run_step(name, cmd)
    print("[DONE] myset pipeline completed successfully.")


if __name__ == "__main__":
    main()
