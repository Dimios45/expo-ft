# Fresh five-episode test: lan-five-20261007-001

Task: `fold the towel`. Five episodes, 300 seconds each. Initial policy is the
existing molmoact2-yam-pi05 JAX base, bf16, with an empty new learner registry.
No old EXPO checkpoints, replay or labels are imported. Configuration was saved
on all three hosts; the model/learner starts only when the operator runs below.

Finish the separate active robot/GPU job before launching. No service is killed
automatically. The launchers reject occupied service ports, existing GPU compute
processes, and competing KARMA controllers.

Run in order:

4090:
```bash
cd /home/sra/tirth/expo-ft
python3 scripts/yam/five_episode_test.py 4090
```
Wait for `Online learner ready` (first startup fingerprints the model).

3060:
```bash
python3 /home/sra/yam-lan/five_episode_test.py 3060
```
Wait for `3060 warmed; online policy ready`.

NUC:
```bash
python3 /home/yambox/yam-lan/five_episode_test.py nuc
```

These commands attach to named tmux sessions. Rerun the same command to reattach.
Detach with Ctrl+B then D. Closing SSH or the terminal does not terminate the
session and does NOT stop an active robot episode. Stay physically available;
use Ctrl+C in the NUC session for KARMA's normal stop/parking/cleanup, and keep the
physical stop available. Killing tmux/the host or losing power is not graceful
shutdown. Reboots still require inspection of any pending episode/learner state.

At each episode boundary type `ready`. KARMA then displays the fixed task and
requires Enter before power-up. Random nonempty task input is rejected. Reward
labels re-prompt on invalid input: `0`/`1`, then `failure`/`truncated` for zero.
KARMA's own y/n outcome prompt remains. Frames, outcomes and policy versions are
preserved; an interrupted/error take is not automatically admitted as training.

A dead process remains visible in its tmux pane. The command reattaches to show
its error; it does not automatically restart robot control. After diagnosing it,
remove only its stopped session with `tmux kill-session -t NAME` and rerun the
launcher. Names are `yam-lan-five-20261007-001-4090`, `...-3060`, `...-nuc`.
Do not kill a live NUC session as a substitute for Ctrl+C and hardware cleanup.

## Logs retained

Central run root on 4090:
`/home/sra/yam-online-lan/lan-five-20261007-001/`

- `test.json`: pinned task, count, duration and fresh-base intent.
- `logs/`: 4090 tmux terminal transcript.
- `monitoring/progress.log` and `progress.jsonl`: automatic five-minute progress,
  outcomes, policy versions, queues and current stage log tails.
- `monitoring/remote-logs/{nuc,inference}/`: five-minute copies of remote console
  logs. Original logs also remain on the remote hosts if synchronization fails.
- `monitoring/learner-logs/`: copies of this run's stage logs and GPU timing reports.
- `learner/online/`: resumable uploads, admitted episodes, jobs and learner status.
- `learner/versions/`: immutable trained checkpoints and manifests.

NUC originals: `~/yam-expo-data/lan-five-20261007-001/` for datasets, per-episode
`runtime-logs/`, labels, upload receipts and final summary; and
`~/yam-lan/runs/lan-five-20261007-001/logs/` for tmux console output.
3060 originals: `~/yam-online-lan/lan-five-20261007-001/logs/`.

A successful five-episode collection drains uploads/training and writes its final
summary. The monitor then stops after its final snapshot/log sync. A paused or
failed run remains monitored. Neither terminal persistence nor logging provides
an uninterrupted control deadline guarantee.

Validation: 21 targeted tests passed; shell/Python syntax checks passed; a detached
tmux output-capture smoke test passed. Remote launcher copies and the pinned test
configuration were verified. No new model service or robot controller was started
while preparing this test.
