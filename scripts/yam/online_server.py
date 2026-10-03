import argparse
import asyncio
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from expo_ft.yam.online import serve
p=argparse.ArgumentParser()
p.add_argument('--port',type=int,default=8208)
p.add_argument('--root',default=str(ROOT/'artifacts/yam-overlap-test'))
a=p.parse_args()
asyncio.run(serve(a.root,ROOT,a.port))
