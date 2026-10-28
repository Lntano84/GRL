"""No NN or optimizer: preserved partial marker for manager lifecycle qualification."""
from pathlib import Path
import os,sys,json,time
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);tmp=out/'partial.tmp';tmp.write_text(json.dumps(dict(fake=True,pid=os.getpid(),optimizer_steps=0)),encoding='utf8');tmp.replace(out/'partial.json')
while True:time.sleep(.05)
