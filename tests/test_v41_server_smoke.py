#!/usr/bin/env python3
"""Serial HTTP smoke using an existing Q2 model. Coordinate memory before running."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
from urllib.error import URLError


def check_continuation(origin, output):
    messages = [dict(role='user', content='What is 17 + 25? Answer with just the number.')]
    results = []
    for _ in range(2):
        body = dict(messages=messages, temperature=0, max_tokens=16, reasoning_effort='none')
        request = Request(origin + '/v1/chat/completions', data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=300) as response:
            answer = json.load(response)
        assert answer['choices'][0]['finish_reason']
        message = answer['choices'][0]['message']
        assert message['content']
        results.append(answer)
        messages += [message, dict(role='user', content='What is 18 + 24? Answer with just the number.')]
    assert results[1]['usage']['prompt_tokens_details']['cached_tokens'] > 0
    (output / 'continuation.json').write_text(json.dumps(results, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('direction', type=Path)
    parser.add_argument('--port', type=int, default=18049)
    parser.add_argument('--output', type=Path, default=Path('work/v41-server-smoke'))
    args = parser.parse_args()
    if os.environ.get('DS4_LOCK_FILE', '/tmp/ds4.lock') != '/tmp/ds4.lock':
        parser.error('retain the normal /tmp/ds4.lock')
    args.output.mkdir(parents=True, exist_ok=False)
    origin = f'http://127.0.0.1:{args.port}'
    # Refuse to test an unrelated existing listener.
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    results = []
    for name, strength in [('stock', None), ('zero', '0'), ('steered', '1')]:
        command = ['./ds4-server', '-m', str(args.model), '--ssd-streaming',
                   '-c', '512', '--host', '127.0.0.1', '--port', str(args.port)]
        if strength is not None:
            command += ['--dir-steering-file', str(args.direction),
                        '--dir-steering-strength', strength]
        print(f'Starting {name}', flush=True)
        started = time.monotonic()
        with (args.output / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 900
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'{name} exited: {process.returncode}')
                    try:
                        with urlopen(origin + '/v1/models', timeout=2) as response:
                            models = json.load(response)
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('server startup exceeded 900 seconds')
                        time.sleep(1)
                assert models['data']
                if name == 'steered':
                    check_continuation(origin, args.output)
                texts = []
                # Repeating a completed prompt exercises invalidation; then use a new prompt.
                for index, prompt in enumerate([
                    'What is 17 + 25? Answer with just the number.',
                    'What is 17 + 25? Answer with just the number.',
                    'What is 6 * 7? Answer with just the number.'
                ]):
                    body = dict(messages=[dict(role='user', content=prompt)],
                                temperature=0, max_tokens=16, reasoning_effort='none',
                                stream=index == 1)
                    request = Request(origin + '/v1/chat/completions',
                                      data=json.dumps(body).encode(),
                                      headers={'Content-Type': 'application/json'})
                    with urlopen(request, timeout=300) as response:
                        raw = response.read().decode()
                    (args.output / f'{name}-{index}.txt').write_text(raw)
                    if body['stream']:
                        assert 'data: [DONE]' in raw
                        chunks = [json.loads(line[6:]) for line in raw.splitlines()
                                  if line.startswith('data: ') and line != 'data: [DONE]']
                        text = ''.join(c['choices'][0]['delta'].get('content') or ''
                                       for c in chunks if c.get('choices'))
                        assert any(c.get('choices') and c['choices'][0].get('finish_reason')
                                   for c in chunks)
                    else:
                        result = json.loads(raw)
                        assert 'error' not in result and result['choices'][0]['finish_reason']
                        text = result['choices'][0]['message']['content']
                    assert text
                    texts.append(text)
                assert texts[0] == texts[1], 'stream/repeated-prompt parity failed'
                results.append(dict(name=name, texts=texts, seconds=time.monotonic()-started))
                if name == 'zero':
                    assert texts == results[0]['texts'], 'zero differs from stock'
                print(results[-1], flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print('HTTP stock/zero parity, steered generation, streaming and live KV reuse: OK')


if __name__ == '__main__':
    main()
