"""Recompute existing date-family exhaustion and isolated-generator provenance."""
from pathlib import Path
import hashlib,json,re,importlib.metadata as md
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid80';read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    Q=read(ROOT/'outputs/grid66/fresh_scene_qualification.json');A=read(ROOT/'outputs/grid03/archive_index.json')
    dates=set(re.findall(r'2035-\d{2}-\d{2}',json.dumps(A['scenario_files'])))
    exposed=set(Q['historical_exposed_families'])|{x['date'] for x in Q['families']}
    assert len(dates)==52 and len(exposed)==52 and not dates-exposed
    versions={d.metadata['Name']:d.version for d in md.distributions(path=[str(ROOT/'work/grid80/generator_packages')])}
    source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'work/grid80/generator_packages/chronix2grid').rglob('*.py')}
    result=dict(existing_catalogue=sorted(dates),exposed=sorted(exposed),remaining=sorted(dates-exposed),
        provenance=['outputs/grid03/archive_index.json','outputs/grid66/fresh_scene_qualification.json','outputs/grid68/finished.json'],
        versions=versions,official_generator_source_hashes=source_files,
        references=['https://github.com/BDonnot/ChroniX2Grid','https://grid2op.readthedocs.io/en/stable/user/environment.html'],
        original_runtime_modified=False,installation_note='Extra generator dependencies installed --no-deps into work/grid80/generator_packages; PYTHONPATH is not set for controller workers. Initial import race failure retained as generate_attempt01.log.')
    (OUT/'exposure_and_generator_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(catalogue=len(dates),exposed=len(exposed),remaining=len(dates-exposed),isolated_packages=len(versions))))
if __name__=='__main__':main()
