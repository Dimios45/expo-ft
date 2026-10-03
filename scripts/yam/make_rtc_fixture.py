"""Build hardware-free WebSocket inputs from the latest recorded episode."""
import io
import json
from pathlib import Path
import sys

import json_numpy
import msgpack
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from expo_ft.yam.replay import CAMERAS, dataset_to_wire, decode_selected


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    info = json.loads((dataset / 'meta/info.json').read_text())
    session = json.loads((dataset / 'expo_session.json').read_text())
    meta = pq.read_table(dataset / 'meta/episodes').to_pylist()[0]
    rows = pq.read_table(dataset / 'data').to_pylist()
    indices = np.unique(np.linspace(0, len(rows)-1, 5, dtype=int))
    views = {}
    for role, key in CAMERAS.items():
        prefix = 'videos/' + key
        path = dataset / info['video_path'].format(video_key=key,
            chunk_index=meta[prefix+'/chunk_index'], file_index=meta[prefix+'/file_index'])
        frames = np.rint((meta[prefix+'/from_timestamp'] +
            np.array([rows[i]['timestamp'] for i in indices])) * info['fps']).astype(int)
        decoded = decode_selected(path, frames)
        views[role] = [decoded[int(i)] for i in frames]
    fixtures = []
    for n, index in enumerate(indices):
        payload = dict(state=dataset_to_wire(rows[index]['observation.state'], session['dataset_frame']),
                       instruction=session['prompt'], num_steps=10,
                       normalization_tag='yam_dual_molmoact2')
        for role in CAMERAS:
            stream = io.BytesIO()
            Image.fromarray(views[role][n]).save(stream, format='JPEG', quality=90)
            payload[role+'_cam'] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        fixtures.append(dict(frame=int(index), observation_json=json_numpy.dumps(payload).encode()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(msgpack.packb(fixtures, use_bin_type=True))
    print(f'{len(fixtures)} recorded observations saved to {args.output}')


if __name__ == '__main__':
    main()
