#!/usr/bin/env python3
"""Serial real-model SSD CLI stock/zero/effect smoke. Coordinate memory first."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--output', type=Path, default=Path('work/v41-cli-smoke'))
    parser.add_argument('--generation-only', action='store_true',
                        help='compare ordinary greedy stock/zero text on both sites')
    args = parser.parse_args()
    if os.environ.get('DS4_LOCK_FILE', '/tmp/ds4.lock') != '/tmp/ds4.lock':
        parser.error('retain the normal /tmp/ds4.lock for this live smoke')
    if 'DS4_CLI_FORCE_SESSION' in os.environ:
        parser.error('unset DS4_CLI_FORCE_SESSION to exercise ordinary greedy CLI routing')
    args.output.mkdir(parents=True, exist_ok=True)
    for site in ('writer', 'residual'):
        subprocess.run([sys.executable, 'tests/make_v41_cli_direction.py',
                        str(args.output / f'{site}.dir'), '--site', site,
                        '--model-sha256', args.model_sha256], check=True)
    base = ['./ds4', '-m', str(args.model), '--ssd-streaming', '-c', '256',
            '--temp', '0', '--seed', '1', '--raw', '-p', 'Vienna is the capital of']
    if args.generation_only:
        results = []
        stock_text = None
        for name, site in [('stock', None), ('writer-0', 'writer'), ('residual-0', 'residual')]:
            command = base + ['-n', '4']
            if site:
                command += ['--dir-steering-file', str(args.output / f'{site}.dir'),
                            '--dir-steering-strength', '0']
            started = time.monotonic()
            print(f'Starting ordinary generation {name}', flush=True)
            with (args.output / f'generate-{name}.log').open('w') as log:
                process = subprocess.run(command, stdout=subprocess.PIPE, stderr=log)
            (args.output / f'generate-{name}.txt').write_bytes(process.stdout)
            result = dict(name=name, command=command, returncode=process.returncode,
                          seconds=time.monotonic() - started)
            results.append(result)
            (args.output / 'generation-results.json').write_text(json.dumps(results, indent=2))
            if process.returncode:
                raise SystemExit(f'ordinary {name} failed: {process.returncode}')
            if stock_text is None:
                stock_text = process.stdout
            assert process.stdout == stock_text, result
            result['exact_stock_output'] = True
            print(result, flush=True)
        (args.output / 'generation-results.json').write_text(json.dumps(results, indent=2))
        print('V4.1 ordinary CLI greedy zero-output parity: OK')
        return
    results = []
    stock = None
    for name, site, strength in [('stock', None, None), ('writer-0', 'writer', 0),
                                 ('writer-1', 'writer', 1), ('residual-0', 'residual', 0),
                                 ('residual-1', 'residual', 1)]:
        command = base + ['--dump-logits', str(args.output / f'{name}.json')]
        if site:
            command += ['--dir-steering-file', str(args.output / f'{site}.dir'),
                        '--dir-steering-strength', str(strength)]
        print(f'Starting {name}', flush=True)
        started = time.monotonic()
        with (args.output / f'{name}.log').open('w') as log:
            rc = subprocess.run(command, stdout=log, stderr=log).returncode
        result = dict(name=name, command=command, returncode=rc,
                      seconds=time.monotonic() - started)
        results.append(result)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
        if rc:
            raise SystemExit(f'{name} failed ({rc}); see {args.output / (name + ".log")}')
        logits = json.loads((args.output / f'{name}.json').read_text())['logits']
        if stock is None:
            stock = logits
        assert len(logits) == len(stock)
        result['changed_logits'] = sum(a != b for a, b in zip(stock, logits))
        result['max_abs_delta'] = max(abs(a - b) for a, b in zip(stock, logits))
        assert (result['changed_logits'] > 0) == (strength == 1), result
        print(result, flush=True)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    print('V4.1 ordinary CLI SSD parity/effect: OK')


if __name__ == '__main__':
    main()
