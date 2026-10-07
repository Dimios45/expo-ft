#!/usr/bin/env python3
"""Run ON THE NUC, in its configured Karma checkout. This command moves robots.

Only Python stdlib is required here. The assistant does not execute this script.
"""

import argparse
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def health(server):
    with urllib.request.urlopen(
        server.rstrip("/").removesuffix("/act") + "/healthz", timeout=10
    ) as r:
        result = json.load(r)
    if result.get("status") != "ok" or "experiment_id" not in result:
        raise ValueError("Not an EXPO round server")
    return result


def require_saved_episode(folder, returncode):
    if returncode != 0:
        raise RuntimeError('KARMA exited with an error; collection stopped. Inspect local files before labeling or retrying.')
    info_path = folder / 'meta/info.json'
    manifest_path = folder / 'openpi_control_rollouts.json'
    if not info_path.exists() or not manifest_path.exists():
        raise RuntimeError('KARMA did not finalize an episode; no outcome label requested')
    info = json.loads(info_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    saved = [e for e in manifest.get('episodes', []) if e.get('saved')]
    if info.get('total_episodes', 0) != 1 or info.get('total_frames', 0) < 1 or len(saved) != 1:
        raise RuntimeError('KARMA saved no complete episode record; no outcome label requested')


def ask_choice(prompt, allowed):
    while True:
        value=input(prompt).strip().lower()
        if value in allowed: return value
        print('Please enter one of: '+', '.join(sorted(allowed)),flush=True)


def preserve_runtime_logs(folder):
    if not folder.exists(): return
    dest=folder/'runtime-logs';dest.mkdir(exist_ok=True)
    sources=[Path('logs/runtime/rollout.log'), *Path('logs').glob('pi_control_node__follower__*__Yam.log')]
    for source in sources:
        if source.is_file(): shutil.copy2(source,dest/source.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--auto-upload", action="store_true", help="Parent runner manages episode upload")
    p.add_argument("--fixed-prompt", action="store_true", help="Confirm the server task instead of retyping it")
    p.add_argument("karma_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    extra = args.karma_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    reserved = {
        "--root",
        "--server",
        "--episodes",
        "--episode-seconds",
        "--fps",
        "--speed",
        "--chunk-size",
        "--num-steps",
        "--prefetch",
        "--no-prefetch",
        "--repo-id",
        "--policy-frame",
    }
    if any(v.split("=")[0] in reserved for v in extra):
        raise ValueError("Do not override versioned collection/execution settings")
    if args.out.exists():
        raise FileExistsError("Use a new dataset directory for each episode")
    before = health(args.server)
    start = time.time()
    print(
        "Collecting policy version",
        before["policy_version"],
        "task:",
        before["prompt"],
        flush=True,
    )
    print(
        "Use that exact task when Karma prompts. Its y/n label is followed by the authoritative 0/1 label here.",
        flush=True,
    )
    command = [
        "uv",
        "run",
        "karma",
        "rollout",
        "--rig",
        "yam_bimanual",
        "--server",
        args.server,
        "--root",
        str(args.out.resolve()),
        "--repo-id",
        "local/yam-expo-round",
        "--episodes",
        "1",
        "--episode-seconds",
        str(args.seconds),
        "--fps",
        "30",
        "--speed",
        "1",
        "--chunk-size",
        "30",
        "--num-steps",
        "10",
        "--no-prefetch",
        *extra,
    ]
    if args.fixed_prompt:
        adapter = Path(__file__).with_name('karma_episode.py')
        if not adapter.is_file(): raise FileNotFoundError(adapter)
        command = ['uv', 'run', '--no-sync', 'python', str(adapter), '--prompt', before['prompt'], '--', *command[3:]]
    process = subprocess.Popen(command)
    while True:
        try:
            returncode = process.wait()
            break
        except KeyboardInterrupt:
            print(
                "Waiting for Karma to finish shutdown and saving; answer its outcome prompts...", flush=True
            )
    try: preserve_runtime_logs(args.out)
    except OSError as exc: print(f'Runtime log copy failed: {exc}',flush=True)
    require_saved_episode(args.out, returncode)
    after = health(args.server)
    for key in ("experiment_id", "policy_version", "session_id"):
        if before[key] != after[key]:
            raise RuntimeError("Server changed during episode; recording not admitted")
    label = ask_choice('Task success reward [0/1]: ', {'0','1'})
    terminal = 'success' if label=='1' else ask_choice(
        'Failure or interrupted/time-limit truncation [failure/truncated]: ', {'failure','truncated'})
    session = {
        "experiment_id": before["experiment_id"],
        "policy_version": before["policy_version"],
        "server_session": before["session_id"],
        "prompt": before["prompt"],
        "reward": int(label),
        "terminal": terminal,
        "dataset_frame": "karma-recorded",
        "prefetch": False,
        "collection_started": start,
        "collection_finished": time.time(),
        "client_returncode": returncode,
        "server_metadata": before,
    }
    (args.out / "expo_session.json").write_text(json.dumps(session, indent=2) + "\n")
    print(
        "Saved versioned episode:",
        args.out,
        "The online runner will upload this episode automatically." if args.auto_upload
        else "Transfer this entire folder to the GPU host before ingest/train.",
    )


if __name__ == "__main__":
    main()
