#!/usr/bin/env python3
"""Pin the experiment task while retaining KARMA's CLI and hardware lifecycle.

Run inside KARMA's environment. Only its task-entry callback is adapted; the
operator still confirms before power_up. No stdin piping or detached controller.
"""
import argparse
import sys


def task_confirmation(prompt, input_fn):
    while True:
        answer = input_fn(
            f"Experiment task: {prompt!r}. Press Enter to start robot control; Ctrl+C cancels: "
        ).strip()
        if not answer:
            return prompt
        print('Task is fixed by the experiment. Press Enter only, or Ctrl+C to cancel.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('karma_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.prompt.strip(): parser.error('Empty experiment prompt')
    argv = args.karma_args[1:] if args.karma_args[:1] == ['--'] else args.karma_args
    if not argv or argv[0] != 'rollout': parser.error('Only rollout is supported')
    from openpi_control import cli
    if not callable(getattr(cli, '_rollout_prompt', None)):
        raise RuntimeError('KARMA task callback changed; inspect compatibility before running')
    cli._rollout_prompt = lambda input_fn, index, total: task_confirmation(args.prompt, input_fn)
    sys.argv = ['karma', *argv]
    return cli.main()


if __name__ == '__main__':
    raise SystemExit(main())
