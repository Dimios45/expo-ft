import asyncio

import msgpack
import pytest

from expo_ft.yam.stream import Journal, handle


def frame(seq, kind, **kwargs):
    return msgpack.packb(dict(schema=1, episode="test", seq=seq, kind=kind, **kwargs),
                         use_bin_type=True)


def test_reconnect_retry_and_restart(tmp_path):
    path = tmp_path / "journal.sqlite3"
    journal = Journal(path)
    assert not journal.append(frame(0, "begin"))["duplicate"]
    assert Journal(path).append(frame(0, "begin"))["duplicate"]
    with pytest.raises(ValueError, match="different bytes"):
        journal.append(frame(0, "begin", changed=True))
    with pytest.raises(ValueError, match="contiguous"):
        journal.append(frame(2, "execution"))
    journal.append(frame(1, "execution"))
    end = frame(2, "end", terminal="truncated", reward=0)
    journal.append(end)
    assert Journal(path).append(end)["duplicate"]
    with pytest.raises(ValueError, match="contiguous"):
        journal.append(frame(3, "observation"))


def test_reject_incomplete_label_without_consuming_sequence(tmp_path):
    journal = Journal(tmp_path / "journal.sqlite3")
    journal.append(frame(0, "begin"))
    with pytest.raises(ValueError):
        journal.append(frame(1, "end", terminal="failure"))
    assert not journal.append(frame(1, "end", terminal="success", reward=1))["duplicate"]


def test_websocket_roundtrip_and_retry(tmp_path):
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect

    async def run():
        journal = Journal(tmp_path / "journal.sqlite3")
        async with serve(lambda ws: handle(ws, journal), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            for duplicate in (False, True):
                async with connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(frame(0, "begin"))
                    ack = msgpack.unpackb(await ws.recv(), raw=False)
                    assert ack["duplicate"] == duplicate
                    assert ack["training_admitted"] is False
                    await ws.send(b"bad frame")
                    assert msgpack.unpackb(await ws.recv(), raw=False)["kind"] == "error"
    asyncio.run(run())


def test_recorded_inference_bridge(tmp_path, monkeypatch):
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
    from expo_ft.yam import stream

    def mock_infer(body):
        assert body == b'recorded input'
        return {'actions': [[0.0] * 14] * 30, 'policy_reported_ms': 1.0}

    monkeypatch.setattr(stream, 'infer_recorded', mock_infer)

    async def run():
        journal = Journal(tmp_path / 'journal.sqlite3')
        lock = asyncio.Lock()
        event = tmp_path / 'live-event.json'
        async with serve(lambda ws: handle(ws, journal, lock, event), '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}') as ws:
                raw = msgpack.packb(dict(schema=1, kind='infer_recorded', seq=7,
                    observation_json=b'recorded input'), use_bin_type=True)
                await ws.send(raw)
                response = msgpack.unpackb(await ws.recv(), raw=False)
                assert response['kind'] == 'inference_result'
                assert response['seq'] == 7
                assert len(response['actions']) == 30
                assert not event.exists()
                async with lock:
                    await ws.send(raw)
                    response = msgpack.unpackb(await ws.recv(), raw=False)
                    assert response['kind'] == 'error'
                    assert 'busy' in response['message']
                raw = msgpack.packb(dict(schema=1, kind='infer_live', seq=8,
                    observation_json=b'recorded input'), use_bin_type=True)
                await ws.send(raw)
                response = msgpack.unpackb(await ws.recv(), raw=False)
                assert response['kind'] == 'inference_result'
                assert event.exists()
    asyncio.run(run())
