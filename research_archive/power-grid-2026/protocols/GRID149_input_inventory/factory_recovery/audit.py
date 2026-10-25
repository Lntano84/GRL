"""Native neutral-command check; no environment or simulator."""
from pathlib import Path
import json,shutil,warnings
import run as D
np=D.np
def main():
 O,R=D.O,D.R
 warnings.filterwarnings('ignore',message='Numba cannot be loaded.*')
 import grid2op
 from grid2op.Action import ActionSpace
 from grid2op.Environment import BaseEnv
 from grid2op.Observation import BaseObservation
 def forbidden(*a,**kw):raise AssertionError('No physical/environment call authorized')
 grid2op.make=BaseEnv.__init__=BaseEnv.reset=BaseEnv.step=BaseObservation.simulate=forbidden
 space=ActionSpace.from_dict(D.read(R/'outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json'))
 av=lambda a:np.concatenate([a._get_array_from_attr_name(k).astype(np.float32) for k in type(a).attr_list_vect])
 neutral=space();assert np.all(neutral._curtail==-1)
 for k in type(neutral).attr_list_vect:
  if k!='_curtail':assert np.all(neutral._get_array_from_attr_name(k)==0)
 inv=D.read(R/'outputs/grid147_generalization_diagnosis/saved_state_inventory.json')['records'];empty=0;status=0;linevalues=set()
 for r in inv:
  with np.load(R/r['observation_file']) as z:
   a=space.from_vect(z['base_action']);assert np.array_equal(av(a),z['base_action'])
   assert all(np.array_equal(a._get_array_from_attr_name(k),neutral._get_array_from_attr_name(k)) for k in type(a).attr_list_vect if k!='_set_line_status')
   v=a._set_line_status;linevalues|=set(int(x) for x in v if x!=0)
   if np.any(v!=0):status+=1
   else:empty+=1;assert np.array_equal(av(a),av(neutral))
 assert empty==229 and status==76 and linevalues=={1}
 D.save('native_command_audit.json',dict(passed=True,neutral_defaults_verified=True,base_vector_roundtrips=305,no_op=empty,only_line_reconnection=status,other_commands=0,
  conclusion='Missing continuous redispatch/curtail/storage commands do not add variation to these saved search inputs. Candidate feedback already includes the shared reconnection context;explicit reconnect locations remain a possible public input,not proven useful.',new_physical_calls=0,new_fits=0))
 p=O/'GRID149_report.md';p.write_text(p.read_text(encoding='utf8')+'\n## 基础命令原生复核\n\n用原生序列化动作空间核对默认空动作及305次向量往返：229次为空动作，76次只有线路重连（状态命令+1）；调度、储能、限发及拓扑命令均与空动作一致。因此，这批搜索中没有被遗漏的连续基础控制变化可供学习，不能把它当作下一特征扩展的依据。线路重连位置本身仍可能作为公开上下文，但尚未建立预测价值。没有构造环境或发送仿真。\n',encoding='utf8',newline='\n')
 c=R/'论文协作资料/FRAME_输入目标对齐_GRID149'
 for n in ['native_command_audit.json','GRID149_report.md']:shutil.copy2(O/n,c/n)
 d=D.A/'native_audit';d.mkdir();
 for p in [Path(__file__),O/'native_command_audit.json',O/'GRID149_report.md']:shutil.copy2(p,d/p.name)
 (d/'archive_manifest.json').write_text(json.dumps({p.name:D.sha(p) for p in d.iterdir()},ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
 for p,h in D.read(O/'source_hashes.json').items():assert D.sha(R/p)==h
 for p,h in D.read(O/'state_sources.json').items():assert D.sha(R/p)==h
 n=0
 for phase in D.A.iterdir():
  for p,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/p)==h;n+=1
 for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
 D.save('publication_verification.json',dict(passed=True,archive_files_verified=n,collaboration_files_verified=len(list(c.iterdir())),zero_fits=True))
 d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json')
 (d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
 print('GRID149_NATIVE_AUDIT',empty,status)
if __name__=='__main__':main()
