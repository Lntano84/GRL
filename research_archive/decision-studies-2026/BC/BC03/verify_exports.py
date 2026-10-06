"""Independent arithmetic audit of exported trajectories, without another replay."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import *

def main():
    out = BASE/'outputs'
    audit = json.loads((out/'BC03_audit.json').read_text())
    results = {}
    rows = list(csv.DictReader((out/'BC03_all_windows.csv').open(encoding='utf-8-sig',newline='')))
    assert len(rows) == len(audit['summaries'])*1008
    for arm, summary in audit['summaries'].items():
        group = [r for r in rows if r['method']==arm]
        assert len(group)==1008 and [int(r['window_index']) for r in group]==list(range(1008))
        evaluation = [r for r in group if r['phase']=='evaluation']
        assert len(evaluation)==864 and int(evaluation[0]['window_index'])==144
        assert float(evaluation[0]['trace_start_s'])==EVAL_START
        max_dt_error = 0.
        for row in group:
            duration = float(row['duration_s'])
            # Independently write the unit conversion: 0.1%=0.001, 36 disks.
            cost = float(row['get_service_time_s'])+float(row['put_service_time_s'])
            calculated = cost/(36*.001*duration)*100
            max_dt_error = max(max_dt_error,abs(calculated-float(row['dt_total_pct'])))
            assert int(row['flash_write_bytes_sample'])==int(row['flash_write_chunks'])*131072
            assert int(row['flash_write_chunks'])==int(row['flash_prefetch_write_chunks'])+int(row['flash_other_write_chunks'])
            assert int(row['backend_prefetch_read_bytes_sample'])==int(row['backend_prefetch_read_chunks'])*131072
        assert max_dt_error <= 1e-8
        peak = max(evaluation,key=lambda r:float(r['dt_total_pct']))
        avg = math.fsum(float(r['dt_total_pct'])*float(r['duration_s']) for r in evaluation)/math.fsum(float(r['duration_s']) for r in evaluation)
        writes = sum(int(r['flash_write_chunks']) for r in evaluation)
        reads = sum(int(r['backend_prefetch_read_chunks']) for r in evaluation)
        assert writes==summary['evaluation_write_chunks']
        assert reads==summary['evaluation_prefetch_read_chunks']
        assert int(peak['window_index'])==summary['peak_window']
        assert abs(float(peak['dt_total_pct'])-summary['peak_dt_pct'])<=1e-8
        assert abs(avg-summary['average_dt_pct'])<=1e-8
        assert (writes<=148731)==summary['resource_eligible']
        results[arm]={'max_dt_conversion_error_pp':max_dt_error,'average_recalculation_error_pp':abs(avg-summary['average_dt_pct']),
                      'rows':len(group),'evaluation_rows':len(evaluation),'peak_window':int(peak['window_index'])}
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('BC03*') if p.is_file()}
    verification={'passed':True,'checks':results,'deliverables_sha256':hashes}
    (WORK/'export_verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps({'export_verification_passed':True,'checks':results}))

if __name__=='__main__':
    main()
