"""Structural qualification of selected chronics; no model or agent outcomes."""
import bz2
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/grid03"
DATA=ROOT/"work/grid03/formal_env"
split=json.loads((OUT/"scenario_split_frozen.json").read_text(encoding="utf-8"))
start=time.monotonic()
rows=[]
required=["load_p.csv.bz2","load_q.csv.bz2","prod_p.csv.bz2","load_p_forecasted.csv.bz2","load_q_forecasted.csv.bz2","prod_p_forecasted.csv.bz2","start_datetime.info","time_interval.info"]
for fold,names in split["extracted_subsets"].items():
    for name in names:
        d=DATA/"chronics"/name
        assert all((d/f).is_file() for f in required)
        info={"split":fold,"scenario":name,"files":{},"start_datetime":(d/"start_datetime.info").read_text().strip(),"time_interval":(d/"time_interval.info").read_text().strip()}
        assert info["time_interval"] in ["00:05","0:05","00:05:00"]
        for f in [x for x in required if x.endswith("bz2")]:
            with bz2.open(d/f,"rt",encoding="utf-8",newline="") as stream:
                reader=csv.reader(stream,delimiter=";")
                header=next(reader)
                # Count records and check widths; do not compute load/quality statistics or select by them.
                count=0
                for row in reader:
                    assert len(row)==len(header)
                    count+=1
            expected_cols=62 if f.startswith("prod_p") else 99
            assert len(header)==expected_cols and len(set(header))==expected_cols
            info["files"][f]={"rows":count,"columns":len(header),"header_sha256":hashlib.sha256(";".join(header).encode()).hexdigest()}
        counts=info["files"]
        actual=[counts[x]["rows"] for x in ["load_p.csv.bz2","load_q.csv.bz2","prod_p.csv.bz2"]]
        forecast=[counts[x]["rows"] for x in ["load_p_forecasted.csv.bz2","load_q_forecasted.csv.bz2","prod_p_forecasted.csv.bz2"]]
        assert len(set(actual))==1 and len(set(forecast))==1
        assert actual[0]>=2016 and forecast[0]==12*actual[0]
        rows.append(info)
        print(json.dumps({"checked":len(rows),"scenario":name,"rows":actual[0],"forecast_rows":forecast[0],"wall_s":round(time.monotonic()-start,1)}),flush=True)
result={"passed":True,"count":len(rows),"physical_steps":0,"schema":rows,"wall_s":time.monotonic()-start,
        "scope":"Required files, interval, header uniqueness, row widths/counts; not AC feasibility or all-value quality certification."}
(OUT/"schema_check.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
