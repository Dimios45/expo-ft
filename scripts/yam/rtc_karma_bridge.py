#!/usr/bin/env python3
"""NUC localhost HTTP adapter to persistent WebSocket. No direct robot access.

KARMA keeps its normal HTTP client, checks, and executor. No request is retried
automatically; timeout/disconnect closes the socket and returns an HTTP error.
"""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import time

import msgpack
from websockets.sync.client import connect


class Bridge:
    def __init__(self, url, timeout):
        self.url, self.timeout = url, timeout
        self.socket = None
        self.seq = 0

    def call(self, kind, **fields):
        started = time.monotonic()
        try:
            if self.socket is None:
                self.socket = connect(self.url, compression=None, open_timeout=self.timeout,
                                      max_size=8*1024*1024)
            self.seq += 1
            self.socket.send(msgpack.packb(dict(schema=1, kind=kind, seq=self.seq, **fields),
                                           use_bin_type=True))
            remaining = self.timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("request deadline exceeded")
            response = msgpack.unpackb(self.socket.recv(timeout=remaining), raw=False)
            if time.monotonic() - started > self.timeout:
                raise TimeoutError("late response discarded")
            if response.get('kind') == 'error' or response.get('seq') != self.seq:
                raise ValueError(f"WebSocket request failed: {response}")
            return response
        except Exception:
            if self.socket is not None:
                self.socket.close()
                self.socket = None
            raise


def handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path not in ('/act', '/healthz'):
                return self.reply(404, {'error': 'unknown path'})
            try:
                health = bridge.call('health')['health']
                health.update(transport='websocket_via_nuc_bridge', rtc_enabled=False)
                self.reply(200, health)
            except Exception as exc:
                self.reply(502, {'error': str(exc)})

        def do_POST(self):
            if self.path != '/act':
                return self.reply(404, {'error': 'unknown path'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 7*1024*1024:
                    raise ValueError('invalid payload size')
                self.connection.settimeout(10)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('incomplete payload')
                result = bridge.call('infer_live', observation_json=body)
                actions = result['actions']
                if len(actions) != 30 or any(len(row) != 14 or
                    any(not math.isfinite(v) for v in row) for row in actions):
                    raise ValueError('invalid action shape or values')
                self.reply(200, {'actions': actions, 'dt_ms': result['policy_reported_ms']})
            except Exception as exc:
                self.reply(502, {'error': str(exc)})
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:8205')
    parser.add_argument('--port', type=int, default=8207)
    parser.add_argument('--timeout', type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.timeout <= 30:
        parser.error('timeout must be between 0 and 30 seconds')
    print(f'KARMA bridge http://127.0.0.1:{args.port} -> {args.url}', flush=True)
    HTTPServer(('127.0.0.1', args.port), handler(Bridge(args.url, args.timeout))).serve_forever()
