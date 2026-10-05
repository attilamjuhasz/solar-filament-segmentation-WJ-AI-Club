from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

VERSION = "0.3.0"
PROTECTED_DEFAULT = [
    "src/evaluation/competition_adapter.py",
    "src/data/folds.py",
    "tests/test_metric_parity.py",
    "tests/test_fold_isolation.py",
    "validate_submission.py",
]

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "lane": {"type": "string", "enum": ["postprocess", "inference", "loss", "augmentation", "training", "architecture", "bugfix"]},
        "hypothesis": {"type": "string"},
        "rationale": {"type": "string"},
        "files_allowed": {"type": "array", "items": {"type": "string"}},
        "expected_effect": {"type": "string"},
        "estimated_minutes": {"type": "integer", "minimum": 1, "maximum": 90},
        "risk": {"type": "string"},
        "validation_plan": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "lane", "hypothesis", "rationale", "files_allowed", "expected_effect", "estimated_minutes", "risk", "validation_plan"],
    "additionalProperties": False,
}

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "fatal_flaws": {"type": "array", "items": {"type": "string"}},
        "recommended_changes": {"type": "array", "items": {"type": "string"}},
        "leakage_or_overfit_risk": {"type": "string"},
    },
    "required": ["approve", "fatal_flaws", "recommended_changes", "leakage_or_overfit_risk"],
    "additionalProperties": False,
}

ENGINEER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "known_risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "files_changed", "tests_run", "known_risks"],
    "additionalProperties": False,
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str] | str, cwd: Path | None = None, timeout: int | None = None,
        shell: bool = False, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True,
                          timeout=timeout, shell=shell, input=input_text)


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    p = run(["git", *args], cwd=repo)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def flatten_numbers(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numbers(v, name))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and prefix:
        out[prefix] = float(obj)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_snapshot(repo: Path, rels: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in rels:
        p = repo / rel
        if not p.is_file():
            raise FileNotFoundError(f"Protected file missing: {rel}")
        result[rel] = sha256(p)
    return result


def verify_protected(repo: Path, snap: dict[str, str]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    for rel, digest in snap.items():
        p = repo / rel
        if not p.is_file() or sha256(p) != digest:
            bad.append(rel)
    return not bad, bad


class Ledger:
    def __init__(self, repo: Path):
        self.root = repo / ".autoresearch"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "events.jsonl"
        self.state_file = self.root / "state.json"

    def state(self) -> dict:
        return load_json(self.state_file, {}) or {}

    def set_state(self, state: dict) -> None:
        save_json(self.state_file, state)

    def append(self, kind: str, payload: dict) -> None:
        row = {"time": utcnow(), "kind": kind, "payload": payload}
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def recent(self, n: int = 30) -> list[dict]:
        if not self.events.exists():
            return []
        out = []
        for line in self.events.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


class Codex:
    def __init__(self):
        self.exe = shutil.which("codex")
        if not self.exe:
            raise RuntimeError("Codex CLI is not installed or not on PATH.")

    def structured(self, prompt: str, schema: dict, cwd: Path, out: Path,
                   sandbox: str, effort: str, timeout: int) -> dict:
        out.parent.mkdir(parents=True, exist_ok=True)
        schema_path = out.with_suffix(".schema.json")
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        cmd = [
            self.exe, "exec", "--ephemeral", "--cd", str(cwd),
            "--sandbox", sandbox, "--ask-for-approval", "never",
            "--config", f"model_reasoning_effort={effort}",
            "--output-schema", str(schema_path), "--output-last-message", str(out), "-"
        ]
        p = run(cmd, cwd=cwd, timeout=timeout, input_text=prompt)
        out.with_suffix(".codex.log").write_text(
            f"STDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n", encoding="utf-8", errors="replace")
        if p.returncode:
            raise RuntimeError(f"Codex failed with exit code {p.returncode}")
        return json.loads(out.read_text(encoding="utf-8"))


def hardware() -> dict:
    info = {"platform": platform.platform()}
    try:
        p = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=10)
        if p.returncode == 0:
            info["gpu"] = p.stdout.strip()
    except Exception:
        info["gpu"] = "unknown"
    return info


def proposal_prompt(state: dict, recent: list[dict]) -> str:
    return f"""You are the Researcher in an autonomous Kaggle CV lab.
Competition: Solar Filament Segmentation 2026.
Goal: improve strict local generalization, prioritizing strict PQ, without benchmark leakage.
Hardware: {json.dumps(hardware(), indent=2)}
Current champion: {json.dumps(state, indent=2)}
Recent history: {json.dumps(recent[-16:], indent=2)}

Propose exactly ONE falsifiable experiment with high expected information gain per minute.
Prefer postprocess/inference/bugfix experiments before expensive retraining unless evidence strongly favors training.
Do not alter folds, metric definitions, protected tests, or submission validators.
Do not hardcode test IDs, test masks, or leaderboard-specific behavior.
Assume an 8GB laptop GPU: avoid giant models and unbounded sweeps.
The eventual engineer must create .autoresearch_candidate.json.
Return only schema-compliant JSON."""


def critic_prompt(proposal: dict, protected: list[str]) -> str:
    return f"""You are an independent skeptical critic. Do not edit files.
Proposal:\n{json.dumps(proposal, indent=2)}
Protected files:\n{json.dumps(protected, indent=2)}
Reject if it leaks the sealed confirmation split into the inner loop, changes metrics/folds, weakens tests, hardcodes test outputs, is too broad to attribute, or wastes compute relative to likely gain.
Return only schema-compliant JSON."""


def engineer_prompt(proposal: dict, state: dict, protected: list[str], minutes: int) -> str:
    return f"""You are the Engineer inside an isolated Git worktree.
Approved experiment:\n{json.dumps(proposal, indent=2)}
Champion state:\n{json.dumps(state, indent=2)}
Protected files:\n{json.dumps(protected, indent=2)}

Implement only this experiment. Do not change protected files, folds, metric definitions, or validation rules. Do not hardcode test IDs or outputs. Keep compute under about {minutes} minutes.
You may run tests, bounded training, and local evaluation.
Before finishing, create `.autoresearch_candidate.json` with exactly:
{{
  "kind": "checkpoint" or "postprocess",
  "checkpoint": "relative/path/to/checkpoint.pt",
  "eval_overrides": {{ optional inference overrides }},
  "notes": "short description"
}}
For a postprocess-only experiment, reuse the champion checkpoint and encode only the changed overrides. For a training/model/loss experiment, point to the newly trained checkpoint.
Do not commit; the controller owns Git commits.
Return only schema-compliant JSON."""


def eval_candidate(worktree: Path, split: str, output: Path, timeout: int) -> dict:
    manifest = load_json(worktree / ".autoresearch_candidate.json")
    if not isinstance(manifest, dict):
        raise RuntimeError("Candidate manifest missing.")
    ckpt = Path(str(manifest.get("checkpoint", "")))
    if not ckpt.is_absolute():
        ckpt = worktree / ckpt
    if not ckpt.is_file():
        raise RuntimeError(f"Candidate checkpoint does not exist: {ckpt}")
    allowed = {
        "method", "high_threshold", "low_threshold", "center_threshold", "boundary_weight",
        "marker_min_distance", "max_peaks", "max_instances", "min_area", "tile_size", "stride", "tile_batch_size"
    }
    overrides = {k: v for k, v in (manifest.get("eval_overrides") or {}).items() if k in allowed}
    args = [sys.executable, "evaluate.py", "--checkpoint", str(ckpt), "--fold", "0", "--split", split, "--device", "cuda"]
    flag_map = {
        "marker_min_distance": "--min-distance",
        "max_instances": "--max-instances",
        "max_peaks": "--max-peaks",
        "min_area": "--min-area",
        "high_threshold": "--high-threshold",
        "low_threshold": "--low-threshold",
        "center_threshold": "--center-threshold",
        "boundary_weight": "--boundary-weight",
        "tile_size": "--tile-size",
        "stride": "--stride",
        "tile_batch_size": "--tile-batch-size",
        "method": "--method",
    }
    for k, v in overrides.items():
        args += [flag_map[k], str(v)]
    before = set((worktree / "artifacts" / "reports").glob("*.json")) if (worktree / "artifacts" / "reports").exists() else set()
    p = run(args, cwd=worktree, timeout=timeout)
    log = output.with_suffix(".log.txt")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"STDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n", encoding="utf-8", errors="replace")
    if p.returncode:
        raise RuntimeError(f"Evaluation failed; see {log}")
    after = set((worktree / "artifacts" / "reports").glob("*.json"))
    new_files = sorted(after - before, key=lambda x: x.stat().st_mtime)
    if not new_files:
        candidates = sorted(after, key=lambda x: x.stat().st_mtime)
        if not candidates:
            raise RuntimeError("evaluate.py produced no JSON report.")
        report_path = candidates[-1]
    else:
        report_path = new_files[-1]
    report = load_json(report_path)
    save_json(output, report)
    return report


def decide(old: dict[str, float], new: dict[str, float], min_delta: float = 0.002, max_dice_regression: float = 0.01) -> tuple[bool, str]:
    pq = "overall.pq"
    dice = "overall.mean_dice"
    if pq not in old or pq not in new:
        return False, "strict PQ missing from old/new metrics"
    gain = new[pq] - old[pq]
    if gain < min_delta:
        return False, f"PQ gain {gain:.4f} < required {min_delta:.4f}"
    if dice in old and dice in new and old[dice] - new[dice] > max_dice_regression:
        return False, f"Dice regression {old[dice]-new[dice]:.4f} too large"
    return True, f"PQ improved by {gain:.4f} without disallowed Dice regression"


def install_self(repo: Path, source_file: Path) -> None:
    target = repo / "autoresearcher_max.py"
    if source_file.resolve() != target.resolve():
        shutil.copy2(source_file, target)
    gi = repo / ".gitignore"
    text = gi.read_text(encoding="utf-8") if gi.exists() else ""
    marker = "# AutoResearcher local state"
    if marker not in text:
        with gi.open("a", encoding="utf-8") as f:
            f.write("\n# AutoResearcher local state\n/.autoresearch/\n/.autoresearch_candidate.json\n/kaggle_ready/\n")


def initialize(repo: Path, checkpoint: str | None) -> None:
    ledger = Ledger(repo)
    if checkpoint is None:
        preferred = repo / "artifacts/runs/run_20260930_111126_15cb60/checkpoints/epoch_006.pt"
        if preferred.is_file():
            checkpoint = str(preferred.relative_to(repo))
        else:
            pts = sorted((repo / "artifacts").rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if (repo / "artifacts").exists() else []
            if pts:
                checkpoint = str(pts[0].relative_to(repo))
    if not checkpoint:
        train_dir = repo / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images"
        if train_dir.exists():
            print("No champion checkpoint found. Training a local 6-epoch ResNet34 baseline automatically...")
            ptrain = run([sys.executable, "train.py", "--config", "configs/b0_resnet34.yaml", "--fold", "0", "--epochs", "6", "--device", "cuda"], cwd=repo, timeout=6*60*60)
            (ledger.root / "baseline_training.log.txt").write_text(
                f"STDOUT:\n{ptrain.stdout}\n\nSTDERR:\n{ptrain.stderr}\n", encoding="utf-8", errors="replace")
            if ptrain.returncode:
                raise RuntimeError("Automatic baseline training failed; inspect .autoresearch/baseline_training.log.txt")
            pts = sorted((repo / "artifacts").rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                checkpoint = str(pts[0].relative_to(repo)).replace("\\", "/")
        if not checkpoint:
            raise RuntimeError("No champion checkpoint and no usable dataset found. Configure local Kaggle credentials and rerun the installer with -DownloadCompetitionData.")
    ckpt = repo / checkpoint
    if not ckpt.is_file():
        raise RuntimeError(f"Checkpoint missing: {ckpt}")

    temp_manifest = repo / ".autoresearch_candidate.json"
    save_json(temp_manifest, {"kind": "checkpoint", "checkpoint": checkpoint, "eval_overrides": {}, "notes": "initial champion"})
    report_path = ledger.root / "champion_tuning.json"
    report = eval_candidate(repo, "tuning", report_path, timeout=7200)
    state = {
        "version": VERSION,
        "initialized_at": utcnow(),
        "champion_commit": head(repo),
        "champion_checkpoint": checkpoint,
        "champion_manifest": load_json(temp_manifest),
        "champion_metrics": flatten_numbers(report),
        "protected": protected_snapshot(repo, PROTECTED_DEFAULT),
        "iteration": 0,
        "promotions": 0,
    }
    ledger.set_state(state)
    ledger.append("initialized", state)
    print(json.dumps(state, indent=2))


def create_worktree(repo: Path, exp_id: str, parent: str) -> tuple[Path, str]:
    root = repo / ".autoresearch" / "worktrees"
    root.mkdir(parents=True, exist_ok=True)
    path = root / exp_id
    branch = f"exp/{exp_id}"
    git(repo, "worktree", "remove", "--force", str(path), check=False)
    shutil.rmtree(path, ignore_errors=True)
    git(repo, "branch", "-D", branch, check=False)
    git(repo, "worktree", "add", "-b", branch, str(path), parent)
    return path, branch


def cleanup_worktree(repo: Path, path: Path, branch: str, keep: bool) -> None:
    git(repo, "worktree", "remove", "--force", str(path), check=False)
    shutil.rmtree(path, ignore_errors=True)
    if not keep:
        git(repo, "branch", "-D", branch, check=False)


def run_one(repo: Path, max_minutes: int = 45, push: bool = True) -> None:
    ledger = Ledger(repo)
    state = ledger.state()
    if not state:
        raise RuntimeError("Initialize first.")
    codex = Codex()
    iteration = int(state["iteration"]) + 1
    exp_id = f"{iteration:04d}"
    expdir = ledger.root / "experiments" / exp_id
    expdir.mkdir(parents=True, exist_ok=True)

    proposal = codex.structured(proposal_prompt(state, ledger.recent()), PROPOSAL_SCHEMA, repo,
                                expdir / "proposal.json", "read-only", "high", 900)
    ledger.append("proposal", {"id": exp_id, **proposal})
    critique = codex.structured(critic_prompt(proposal, PROTECTED_DEFAULT), CRITIC_SCHEMA, repo,
                                expdir / "critique.json", "read-only", "medium", 600)
    ledger.append("critique", {"id": exp_id, **critique})
    if not critique["approve"]:
        state["iteration"] = iteration
        ledger.set_state(state)
        ledger.append("rejected_before_compute", {"id": exp_id, "critique": critique})
        return

    wt, branch = create_worktree(repo, exp_id, state["champion_commit"])
    keep = False
    try:
        # Champion artifacts are gitignored. Copy the champion checkpoint into the worktree privately.
        champ_src = repo / state["champion_checkpoint"]
        champ_dst = wt / state["champion_checkpoint"]
        champ_dst.parent.mkdir(parents=True, exist_ok=True)
        if champ_src.is_file() and not champ_dst.is_file():
            shutil.copy2(champ_src, champ_dst)
        # Data is gitignored. Junction/copy access via environment symlink when possible.
        src_data = repo / "data"
        dst_data = wt / "data"
        if src_data.exists() and not dst_data.exists():
            try:
                os.symlink(src_data, dst_data, target_is_directory=True)
            except OSError:
                if os.name == "nt":
                    # Directory junction works on normal Windows accounts without Developer Mode.
                    run(["cmd", "/c", "mklink", "/J", str(dst_data), str(src_data)], cwd=wt, timeout=30)

        engineer = codex.structured(engineer_prompt(proposal, state, PROTECTED_DEFAULT,
                                                     min(max_minutes, int(proposal["estimated_minutes"]))),
                                    ENGINEER_SCHEMA, wt, expdir / "engineer.json",
                                    "workspace-write", "high", max(1800, max_minutes * 60 + 600))
        ledger.append("engineered", {"id": exp_id, **engineer})

        ok, changed = verify_protected(wt, state["protected"])
        if not ok:
            ledger.append("rejected_protected", {"id": exp_id, "changed": changed})
            state["iteration"] = iteration
            ledger.set_state(state)
            return
        manifest_path = wt / ".autoresearch_candidate.json"
        if not manifest_path.is_file():
            ledger.append("rejected_missing_manifest", {"id": exp_id})
            state["iteration"] = iteration
            ledger.set_state(state)
            return

        tests = run([sys.executable, "-m", "pytest", "-q", "tests/test_metric_parity.py", "tests/test_submission_roundtrip.py", "tests/test_fold_isolation.py"], cwd=wt, timeout=1200)
        (expdir / "tests.log.txt").write_text(f"STDOUT:\n{tests.stdout}\nSTDERR:\n{tests.stderr}", encoding="utf-8", errors="replace")
        if tests.returncode:
            ledger.append("rejected_tests", {"id": exp_id})
            state["iteration"] = iteration
            ledger.set_state(state)
            return

        tuning_report = eval_candidate(wt, "tuning", expdir / "tuning.json", timeout=7200)
        new_metrics = flatten_numbers(tuning_report)
        promote, reason = decide(state["champion_metrics"], new_metrics)
        ledger.append("decision", {"id": exp_id, "promote": promote, "reason": reason,
                                   "old_pq": state["champion_metrics"].get("overall.pq"),
                                   "new_pq": new_metrics.get("overall.pq")})

        git(wt, "add", "-A")
        if git(wt, "status", "--porcelain").stdout.strip():
            git(wt, "commit", "-m", f"AutoResearch {exp_id}: {proposal['title']}")
            candidate_commit = head(wt)
        else:
            candidate_commit = state["champion_commit"]

        if promote:
            # Copy candidate checkpoint out before removing the worktree if it is gitignored.
            manifest = load_json(manifest_path, {})
            cand_rel = str(manifest.get("checkpoint", state["champion_checkpoint"]))
            cand_src = wt / cand_rel
            if cand_src.is_file():
                store = repo / ".autoresearch" / "champions" / exp_id
                store.mkdir(parents=True, exist_ok=True)
                stored = store / cand_src.name
                shutil.copy2(cand_src, stored)
                cand_rel = str(stored.relative_to(repo)).replace("\\", "/")
                manifest["checkpoint"] = cand_rel
            git(repo, "merge", "--ff-only", candidate_commit)
            save_json(repo / ".autoresearch_candidate.json", manifest)
            state["champion_commit"] = candidate_commit
            state["champion_checkpoint"] = cand_rel
            state["champion_manifest"] = manifest
            state["champion_metrics"] = new_metrics
            state["promotions"] = int(state.get("promotions", 0)) + 1
            keep = True
            ledger.append("promoted", {"id": exp_id, "commit": candidate_commit, "reason": reason})
            if push:
                p = git(repo, "push", "origin", "HEAD:autoresearcher-v1", check=False)
                ledger.append("push", {"id": exp_id, "returncode": p.returncode, "stderr": p.stderr[-1500:]})
            # Sparse sealed check every 3 promotions, logged only.
            if state["promotions"] % 3 == 0:
                save_json(repo / ".autoresearch_candidate.json", manifest)
                try:
                    sealed = eval_candidate(repo, "confirmation", ledger.root / "sealed" / f"{exp_id}.json", timeout=7200)
                    ledger.append("sealed_result", {"id": exp_id, "metrics": flatten_numbers(sealed)})
                except Exception as exc:
                    ledger.append("sealed_error", {"id": exp_id, "error": repr(exc)})
        else:
            ledger.append("rejected", {"id": exp_id, "reason": reason})

    finally:
        cleanup_worktree(repo, wt, branch, keep)
        state["iteration"] = iteration
        ledger.set_state(state)


def keep_awake(enable: bool) -> None:
    if os.name != "nt":
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED if enable else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def autopilot(repo: Path, hours: float, iterations: int, max_minutes: int) -> None:
    ledger = Ledger(repo)
    state = ledger.state()
    if not state:
        raise RuntimeError("Initialize first.")
    start = time.monotonic()
    keep_awake(True)
    try:
        for _ in range(iterations):
            if (time.monotonic() - start) / 3600 >= hours:
                break
            try:
                run_one(repo, max_minutes=max_minutes, push=True)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                ledger.append("iteration_crash", {"error": repr(exc)})
                s = ledger.state()
                s["iteration"] = int(s.get("iteration", 0)) + 1
                ledger.set_state(s)
                time.sleep(8)
    finally:
        keep_awake(False)
    make_report(repo)


def make_report(repo: Path) -> Path:
    ledger = Ledger(repo)
    state = ledger.state()
    rows = ledger.recent(500)
    out = ledger.root / "REPORT.md"
    lines = [
        "# AutoResearcher Report", "",
        f"- Champion commit: `{state.get('champion_commit','?')}`",
        f"- Champion checkpoint: `{state.get('champion_checkpoint','?')}`",
        f"- Iterations: {state.get('iteration',0)}",
        f"- Promotions: {state.get('promotions',0)}",
        f"- Tuning PQ: {state.get('champion_metrics',{}).get('overall.pq','?')}", "",
        "## Recent significant events", ""
    ]
    for row in rows:
        if row["kind"] in {"proposal", "decision", "promoted", "rejected", "rejected_tests", "sealed_result", "iteration_crash"}:
            lines.append(f"- **{row['kind']}** `{row['time']}` — `{json.dumps(row['payload'], sort_keys=True)[:850]}`")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def prepare_kaggle(repo: Path, submit: bool) -> None:
    ledger = Ledger(repo)
    state = ledger.state()
    if not state:
        raise RuntimeError("No champion state.")
    manifest = state.get("champion_manifest") or {"kind": "checkpoint", "checkpoint": state["champion_checkpoint"], "eval_overrides": {}}
    save_json(repo / ".autoresearch_candidate.json", manifest)
    ckpt = repo / str(manifest["checkpoint"])
    if not ckpt.is_file():
        raise RuntimeError(f"Champion checkpoint missing: {ckpt}")
    test_dir = repo / "data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images"
    if not test_dir.exists():
        raise RuntimeError("Competition test data missing. Run bootstrap with data download.")
    out_dir = repo / "kaggle_ready"
    out_dir.mkdir(exist_ok=True)
    csv = out_dir / "submission.csv"
    man = out_dir / "submission.manifest.json"
    args = [sys.executable, "inference.py", "--checkpoint", str(ckpt), "--test_images", str(test_dir),
            "--output_csv", str(csv), "--output_manifest", str(man), "--device", "cuda", "--tile-batch-size", "8"]
    mapping = {"method":"--method","high_threshold":"--high-threshold","low_threshold":"--low-threshold",
               "center_threshold":"--center-threshold","boundary_weight":"--boundary-weight","marker_min_distance":"--min-distance",
               "max_peaks":"--max-peaks","max_instances":"--max-instances","min_area":"--min-area","tile_size":"--tile-size","stride":"--stride"}
    for k, v in (manifest.get("eval_overrides") or {}).items():
        if k in mapping:
            args += [mapping[k], str(v)]
    p = run(args, cwd=repo, timeout=14400)
    (out_dir / "inference.log.txt").write_text(f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}", encoding="utf-8", errors="replace")
    if p.returncode:
        raise RuntimeError("Final inference failed; inspect kaggle_ready/inference.log.txt")
    p2 = run([sys.executable, "validate_submission.py", str(csv), "--test-dir", str(test_dir), "--manifest", str(man)], cwd=repo, timeout=3600)
    (out_dir / "validation.log.txt").write_text(f"STDOUT:\n{p2.stdout}\nSTDERR:\n{p2.stderr}", encoding="utf-8", errors="replace")
    if p2.returncode:
        raise RuntimeError("Final validation failed; inspect kaggle_ready/validation.log.txt")
    save_json(out_dir / "CHAMPION_INFO.json", {"commit": state["champion_commit"], "checkpoint": state["champion_checkpoint"], "generated_at": utcnow()})
    print(f"Kaggle-ready CSV: {csv}")
    if submit:
        p3 = run([sys.executable, "-m", "kaggle", "competitions", "submit", "-c", "filament-segmentation-2026", "-f", str(csv), "-m", f"AutoResearcher {state['champion_commit'][:8]}"], cwd=repo, timeout=600)
        print(p3.stdout)
        if p3.returncode:
            print(p3.stderr, file=sys.stderr)
            raise RuntimeError("Kaggle submit failed.")


def main() -> None:
    ap = argparse.ArgumentParser(description="AutoResearcher MAX")
    ap.add_argument("--repo", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("install-self")
    p.add_argument("--source", required=True)
    p = sub.add_parser("init")
    p.add_argument("--checkpoint", default=None)
    p = sub.add_parser("run-one")
    p.add_argument("--max-minutes", type=int, default=45)
    p = sub.add_parser("run")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--iterations", type=int, default=24)
    p.add_argument("--max-minutes", type=int, default=45)
    sub.add_parser("status")
    sub.add_parser("report")
    p = sub.add_parser("prepare-kaggle")
    p.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if args.cmd == "install-self":
        install_self(repo, Path(args.source))
    elif args.cmd == "init":
        initialize(repo, args.checkpoint)
    elif args.cmd == "run-one":
        run_one(repo, args.max_minutes)
        print(make_report(repo))
    elif args.cmd == "run":
        autopilot(repo, args.hours, args.iterations, args.max_minutes)
        print(make_report(repo))
    elif args.cmd == "status":
        print(json.dumps(Ledger(repo).state(), indent=2))
    elif args.cmd == "report":
        print(make_report(repo))
    elif args.cmd == "prepare-kaggle":
        prepare_kaggle(repo, args.submit)


if __name__ == "__main__":
    main()
