"""Versioned LAN contract. No model or robot dependencies.

All wire state/actions use KARMA YAM radians and 1=open grippers. Timestamps
are NUC monotonic nanoseconds; server clocks are never subtracted from them.
"""
from __future__ import annotations

import hashlib
import json
import re

import msgpack
import numpy as np

SCHEMA = 1
MAX_MESSAGE = 16 * 1024 * 1024
CAMERAS = ('top', 'left', 'right')
KARMA_REVISION = 'b4f06f6d645755e605b6c0aec7c10af3d2c911d6'


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        raise ValueError('invalid identifier')
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid SHA-256')
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f'invalid {name}')
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.name not in ('float32', 'float64', 'uint8', 'int32', 'int64', 'bool'):
            raise ValueError('unsupported array dtype')
        a = np.ascontiguousarray(value)
        return msgpack.ExtType(42, msgpack.packb([a.dtype.str, list(a.shape), a.tobytes()], use_bin_type=True))
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _decode(code, payload):
    if code != 42:
        raise ValueError('unknown array extension')
    dtype, shape, data = msgpack.unpackb(payload, raw=False)
    if dtype not in ('<f4', '<f8', '|u1', '<i4', '<i8', '|b1'):
        raise ValueError('unsupported array dtype')
    if not isinstance(shape, list) or len(shape) > 5 or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError('invalid array shape')
    count = 1
    for x in shape:
        count *= x
    if count * np.dtype(dtype).itemsize != len(data):
        raise ValueError('array byte count mismatch')
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def pack(message):
    data = msgpack.packb(message, default=_encode, use_bin_type=True)
    if len(data) > MAX_MESSAGE:
        raise ValueError('message exceeds 16 MiB')
    return data


def unpack(data):
    if not isinstance(data, bytes) or len(data) > MAX_MESSAGE:
        raise ValueError('expected binary message <= 16 MiB')
    result = msgpack.unpackb(data, raw=False, ext_hook=_decode, strict_map_key=True)
    if not isinstance(result, dict) or result.get('schema') != SCHEMA:
        raise ValueError('unsupported message schema')
    return result


def vector(value, shape, name):
    a = np.asarray(value, dtype=np.float32)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'invalid {name}, expected {shape}')
    return a


def observation(obs):
    integer(obs['tick'], 'tick')
    integer(obs['capture_ns'], 'capture_ns')
    vector(obs['state'], (14,), 'state')
    if set(obs['images']) != set(CAMERAS):
        raise ValueError('three YAM views required')
    for name in CAMERAS:
        integer(obs['frame_ids'][name], 'frame id')
        t = integer(obs['image_ns'][name], 'image timestamp')
        if t > obs['capture_ns']:
            raise ValueError('image timestamp after observation')
        frame = obs['images'][name]
        if isinstance(frame, bytes):
            if not frame or len(frame) > 4 * 1024 * 1024:
                raise ValueError('invalid JPEG length')
        elif not (isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                  and frame.ndim == 3 and frame.shape[-1] == 3):
            raise ValueError('expected JPEG bytes or RGB uint8')
    return obs


def schedule(c, d, candidates):
    if type(c) is not int or c not in (8, 12, 15):
        raise ValueError('execution window must be 8, 12 or 15')
    if type(d) is not int or not 1 <= d <= c:
        raise ValueError('require 1 <= delay <= execution window')
    if type(candidates) is not int or candidates not in (2, 4, 8, 32):
        raise ValueError('unsupported candidate count')


def validate_plan(plan, *, episode, version, tick, now_ns, c):
    if plan['episode'] != episode or plan['version'] != version:
        raise ValueError('wrong episode or policy version')
    if plan['start_tick'] != tick or now_ns > plan['expires_ns']:
        raise ValueError('expired or misaligned plan')
    return vector(plan['actions'], (c, 14), 'action window')
