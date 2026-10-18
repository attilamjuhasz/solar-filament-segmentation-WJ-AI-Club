"""Helios v0.1: local no-AI-cost, tuning-only solar segmentation experiments."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OLD = Path.home() / 'Downloads' / 'temp-kaggle-autoresearch'
CKPT = OLD / 'checkpoints' / 'resnet34_unet_fold0_best.pt'
DATA = ROOT / 'data' / 'filament-segmentation-2026' / 'MAGFiLO_1.0_Kaggle_2026'
OUT = ROOT / 'artifacts' / 'helios'
STATE, LEDGER, REPORT = [OUT / x for x in ('state.json', 'experiments.jsonl', 'REPORT_TO_CHATGPT.txt')]
LOCK = OUT / 'run.lock'
PROTECTED = ('evaluate.py', 'train.py', 'src/data/annotations.py', 'src/data/dataset.py',
             'src/data/folds.py', 'src/data/manifest.py', 'src/evaluation/competition_adapter.py',
             'src/inference/config.py', 'src/inference/engine.py', 'src/inference/instances.py',
             'src/inference/rle.py', 'src/models/__init__.py', 'src/models/resnet_unet.py')
TRIALS = (
    ('BASE', {}), ('H080L060', {'high_threshold': .80, 'low_threshold': .60}),
    ('H090L070', {'high_threshold': .90, 'low_threshold': .70}),
    ('H085L065', {'high_threshold': .85, 'low_threshold': .65}),
    ('AREA200', {'min_area': 200}), ('AREA700', {'min_area': 700}),
    ('CAP30', {'max_instances': 30}),
    ('H080AREA250', {'high_threshold': .80, 'low_threshold': .65, 'min_area': 250}),
)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def log(msg):
    print('[HELIOS] ' + msg, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def event(kind, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    with LEDGER.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'at': stamp(), 'event': kind, **fields}, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def completed():
    if not LEDGER.exists():
        return {}
    rows = [json.loads(x) for x in LEDGER.read_text(encoding='utf-8').splitlines() if x.strip()]
    return {r['name']: r for r in rows if r.get('event') == 'passed'}


def pinned():
    paths = {p: ROOT / p for p in PROTECTED}
    paths['annotations'] = DATA / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    paths['folds_manifest'] = ROOT / 'artifacts' / 'folds_manifest.json'
    paths['checkpoint'] = CKPT
    partition = ROOT / 'artifacts' / 'partitions_migrated_v1.json'
    if partition.exists():
        paths['partitions'] = partition
    return {key: digest(path) for key, path in paths.items()}


def recover_manifest():
    target = ROOT / 'artifacts' / 'folds_manifest.json'
    original = OLD / 'artifacts' / 'folds_manifest.json'
    ann = DATA / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    if not target.exists():
        if not original.is_file():
            raise RuntimeError('Original fold manifest not found. Will NOT generate new folds.')
        if read(original).get('annotations_sha256') != digest(ann):
            raise RuntimeError('Annotation checksum differs from original fold manifest.')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        log('Recovered original frozen fold manifest.')
    for name in ('folds.csv', 'partitions_migrated_v1.json'):
        src, dst = OLD / 'artifacts' / name, ROOT / 'artifacts' / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    return target


def provenance():
    import torch
    from src.data.folds import load_frozen_folds_manifest, get_deterministic_fold_partitions
    from src.data.manifest import canonical_observation_id as canon
    manifest = ROOT / 'artifacts' / 'folds_manifest.json'
    ann = DATA / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    assignments, manifest_sha = load_frozen_folds_manifest(manifest, verify_annotations_path=ann)
    # Only use the checkpoint already run and trusted by the user.
    c = torch.load(CKPT, map_location='cpu', weights_only=False)
    keys = ('fold', 'folds_manifest_sha256', 'train_observations', 'val_observations', 'metric_name')
    if any(c.get(k) is None for k in keys):
        raise RuntimeError('Checkpoint lacks verified provenance metadata: refusing optimization.')
    fold = c['fold']
    if manifest_sha != c['folds_manifest_sha256']:
        raise RuntimeError('Checkpoint and original frozen fold manifest hashes differ.')
    if c.get('eligible_for_promotion') is False or c.get('runtime_config', {}).get('is_diagnostic'):
        raise RuntimeError('Checkpoint marked diagnostic; refusing optimization.')
    train = {canon(k) for k in c['train_observations']}
    val = {canon(k) for k in c['val_observations']}
    expected = {canon(k) for k, v in assignments.items() if v == fold}
    if not train or not val or val != expected or train & expected:
        raise RuntimeError('Training/validation observation isolation failed.')
    parts = get_deterministic_fold_partitions(assignments, target_fold=fold)
    tuning, confirmation, explored = (set(parts[k]) for k in ('tuning','confirmation','explored'))
    if not tuning or not confirmation or tuning & confirmation or tuning & explored or confirmation & explored:
        raise RuntimeError('Tuning/confirmation/explored partitions overlap or are missing.')
    if not (tuning | confirmation | explored) <= val:
        raise RuntimeError('Validation partitions contain non-held-out images.')
    partsha = hashlib.sha256(json.dumps({k: sorted(parts[k]) for k in ('tuning','confirmation','explored')}, sort_keys=True).encode()).hexdigest()
    return {'fold': fold, 'epoch': c.get('epoch'), 'manifest_sha': manifest_sha, 'partition_sha': partsha,
            'training': len(train), 'validation': len(val), 'tuning': len(tuning), 'confirmation': len(confirmation)}


def setup():
    os.chdir(ROOT)
    for path in (CKPT, DATA / 'train' / 'train_images', DATA / 'test' / 'test_images'):
        if not path.exists():
            raise FileNotFoundError(f'Missing input: {path}')
    recover_manifest()
    meta, hashes = provenance(), pinned()
    if STATE.exists():
        prior = read(STATE)
        if prior['protected'] != hashes or prior['metadata']['partition_sha'] != meta['partition_sha']:
            raise RuntimeError('Protected research inputs changed. Stop and review before continuing.')
        log('Existing research state preserved.')
    else:
        write(STATE, {'created': stamp(), 'protected': hashes, 'metadata': meta})
        event('setup', metadata=meta)
    log(f'CHECKS PASSED: train={meta["training"]} val={meta["validation"]} tuning={meta["tuning"]} confirmation={meta["confirmation"]}')
    log('SETUP COMPLETE. No GPU experiments started.')


def guard():
    if not STATE.exists():
        raise RuntimeError('Run setup before experiments.')
    if read(STATE)['protected'] != pinned():
        raise RuntimeError('Protected code/data changed; refusing to run.')


def notification():
    if os.name != 'nt':
        return
    # Best-effort desktop alert; no change to PowerShell execution policy.
    script = '''Add-Type -AssemblyName System.Windows.Forms\nAdd-Type -AssemblyName System.Drawing\n$n=New-Object System.Windows.Forms.NotifyIcon\n$n.Icon=[System.Drawing.SystemIcons]::Information\n$n.Visible=$true\n$n.BalloonTipTitle='Project Helios'\n$n.BalloonTipText='GPT checkpoint ready. Paste your Helios report into ChatGPT.'\n$n.ShowBalloonTip(10000)\nStart-Sleep -Seconds 12\n$n.Dispose()'''
    import base64
    code = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    try:
        subprocess.Popen(['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def checkpoint(reason):
    results = sorted(completed().values(), key=lambda x: x['pq'], reverse=True)
    meta = read(STATE)['metadata'] if STATE.exists() else {}
    lines = ['HELIOS STRATEGIC CHECKPOINT FOR GPT-6', f'Time: {stamp()}', f'Reason: {reason}',
             'Kaggle baseline: about 0.20 PQ (previous score; NOT reevaluated here)',
             f'Frozen fold: {meta.get("fold")}; tuning: {meta.get("tuning")}; confirmation: {meta.get("confirmation")} (untouched)',
             f'Completed trials: {len(results)} / {len(TRIALS)}',
             'Best local tuning PQ: ' + (f'{results[0]["pq"]:.6f} ({results[0]["name"]})' if results else 'NOT MEASURED'),
             '', 'All verified local tuning results:']
    for r in results:
        lines.append(f'{r["name"]}: PQ={r["pq"]:.6f}; TP={r["tp"]}; FP={r["fp"]}; FN={r["fn"]}; params={r["params"]}')
    lines.extend(['', 'These are TUNING scores, not confirmed or Kaggle PQ.',
                  'GPT: review experimental evidence and select the next controlled strategy.',
                  'Do not change frozen folds or protected metric evaluation.'])
    body = '\n'.join(lines) + '\n'
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(body, encoding='utf-8')
    event('checkpoint', reason=reason)
    print('\n' + '=' * 60 + '\n' + body + '=' * 60, flush=True)
    if os.name == 'nt':
        try:
            subprocess.run(['clip.exe'], input=body, text=True, check=True, timeout=10)
            log('Checkpoint copied to clipboard. Paste into ChatGPT.')
        except (OSError, subprocess.SubprocessError):
            log(f'Open and copy report: {REPORT}')
    notification()


def trial(name, params):
    guard()
    args = [sys.executable, str(ROOT / 'evaluate.py'), '--checkpoint', str(CKPT),
            '--split', 'tuning', '--device', 'cuda', '--tile-batch-size', '2']
    for key, val in params.items():
        args.extend(['--' + key.replace('_','-'), str(val)])
    log(f'RUNNING {name}: {params} (held-out tuning set only)')
    event('started', name=name, params=params)
    path = OUT / 'logs' / f'{name}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open('w', encoding='utf-8') as log_file:
        proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                if 'Overall PQ:' in line or 'Detailed evaluation report' in line:
                    print('  ' + line.strip(), flush=True)
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    if code:
        raise RuntimeError(f'{name}: evaluator failed (exit {code}); inspect {path}')
    guard()
    text = path.read_text(encoding='utf-8')
    found = re.findall(r'Detailed evaluation report saved to\s+(.+)', text)
    if not found:
        raise RuntimeError(f'{name}: evaluator did not emit a report')
    source = Path(found[-1].strip())
    if not source.is_absolute():
        source = ROOT / source
    report = read(source)
    if report.get('provenance') != 'verified' or report.get('split') != 'tuning' or report.get('is_limited_diagnostic'):
        raise RuntimeError(f'{name}: evaluator provenance/split not acceptable')
    if report.get('total_evaluated_observations') != read(STATE)['metadata']['tuning']:
        raise RuntimeError(f'{name}: wrong number of evaluated observations')
    for key, value in params.items():
        if report['resolved_inference_config'].get(key) != value:
            raise RuntimeError(f'{name}: parameter override was ignored: {key}')
    stats = report['overall']
    pq = float(stats['pq'])
    if not 0 <= pq <= 1:
        raise RuntimeError(f'{name}: invalid PQ {pq}')
    dest = OUT / 'reports' / f'{name}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    row = {'name': name, 'params': params, 'pq': pq, 'tp': stats['tp'], 'fp': stats['fp'],
           'fn': stats['fn'], 'sq': stats['sq'], 'rq': stats['rq'],
           'duration_sec': round(time.monotonic()-started, 1), 'report': str(dest)}
    event('passed', **row)
    log(f'PASSED {name}: verified tuning PQ={pq:.5f}')


def run(limit):
    os.chdir(ROOT)
    guard()
    if LOCK.exists():
        raise RuntimeError(f'Existing run lock (active or stale): {LOCK}')
    LOCK.write_text(str(os.getpid()), encoding='utf-8')
    try:
        count = 0
        for name, params in TRIALS:
            if name in completed():
                log(f'SKIP {name}: previously completed')
                continue
            if count >= limit:
                break
            trial(name, params)
            count += 1
        checkpoint('Experiment batch completed')
    except KeyboardInterrupt:
        checkpoint('Interrupted safely; previous completed results preserved')
        raise SystemExit(130)
    except Exception as exc:
        checkpoint(f'Experiment failed: {type(exc).__name__}: {exc}')
        raise SystemExit(1)
    finally:
        LOCK.unlink(missing_ok=True)


def status():
    log(f'Completed trials: {len(completed())}/{len(TRIALS)}')
    if completed():
        best = max(completed().values(), key=lambda x: x['pq'])
        log(f'Best local tuning: {best["name"]} PQ={best["pq"]:.6f} (unconfirmed)')
    log(f'Checkpoint report: {REPORT} (exists: {REPORT.exists()})')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup')
    sub.add_parser('status')
    r = sub.add_parser('run')
    r.add_argument('--max-trials', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    os.chdir(ROOT)
    try:
        if args.command == 'setup':
            setup()
        elif args.command == 'status':
            status()
        else:
            run(args.max_trials)
    except Exception as exc:
        log(f'ERROR: {type(exc).__name__}: {exc}')
        log('No score claimed. Paste this error into ChatGPT.')
        sys.exit(1)


if __name__ == '__main__':
    main()
