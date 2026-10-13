"""Make a guide-based author draft; no simulation, labels, or model fitting."""
from pathlib import Path
import csv, json, hashlib, re, shutil
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/grid_icaps_draft_v1'
MAT=ROOT/'outputs/grid_submission_materials_v1'
KIT=HERE/'reference_cache/aaai27/AuthorKit27'
if (OUT/'validation.json').exists(): raise SystemExit('Completed draft exists; use a new version.')
OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def text(name,s): (OUT/name).write_text(s.rstrip()+'\n',encoding='utf-8')
originals=[ROOT/'outputs/grid_paper_v1/manuscript.md',ROOT/'outputs/grid_paper_v2/manuscript.md',MAT/'validation.json',ROOT/'outputs/grid61/finished.json',ROOT/'outputs/grid62/finished.json']
original_hashes={str(p.relative_to(ROOT)):sha(p) for p in originals}
source=(HERE/'manuscript.md').read_text(encoding='utf-8')
rows=list(csv.DictReader((MAT/'tables/main_results_unified.csv').open(encoding='utf-8-sig')))
families=list(csv.DictReader((MAT/'tables/all_families.csv').open(encoding='utf-8-sig')))
expected=[('GRID61','FRAME','macro_control_reduction_full',13.11),('GRID61','FRAME','macro_query_reduction_full',69.37),('GRID61','NO_LOCAL','worst_complete_cost_delta_full',4.810),('GRID61','FRAME','worst_complete_cost_delta_full',.372),('GRID62','FRAME','macro_control_reduction_full',6.09),('GRID62','FRAME','macro_query_reduction_full',83.54)]
checks=[]
for stage,m,k,v in expected:
    r=next(r for r in rows if r['stage']==stage and r['method']==m)
    val=float(r[k])*100
    assert abs(val-v)<.0051,(k,val,v)
    checks.append(dict(stage=stage,method=m,metric=k,source_value_percent=val,printed_value=v))
assert len(rows)==8 and len(families)==28
for r in rows:
    fs=[f for f in families if f['stage']==r['stage'] and f['method']==r['method']]
    vals=[float(f['control_reduction_full']) for f in fs]
    assert abs(sum(vals)/len(vals)-float(r['macro_control_reduction_full']))<1e-12
    q=[float(f['query_reduction_full']) for f in fs if f['query_reduction_full']]
    assert abs(sum(q)/len(q)-float(r['macro_query_reduction_full']))<1e-12
    assert sum(f['quality_pass']=='True' for f in fs)==int(r['quality_families'])
    assert sum(f['complete']=='True' for f in fs)==int(r['completed_families'])
banner='> **作者编辑初稿 / AI-assisted working draft, 2026-10-13.** 目标 ICAPS 2027 长文；尚未投稿。新稿依据 Research-Starter-Kit 的公开写作章节重写，保留既有 V1/V2。当前稿件的实验事实不变，文字需作者独立核对和重写；不是已认证符合当届 AI 使用政策的提交稿。\n\n'
md_image='\n\n![Figure 1: conditional substation completion]('+str(OUT/'figures/framework_preview.png').replace('\\','/')+')'
text('manuscript_en.md',banner+source.replace('Figure 1 summarizes the distinction.','Figure 1 summarizes the distinction.'+md_image))
(OUT/'figures').mkdir(exist_ok=True)
for f in ['framework.svg','framework_preview.png','fresh_confirmation.pdf','fresh_confirmation.png']:
    shutil.copy2(MAT/'figures'/f,OUT/'figures'/f)
(OUT/'tables').mkdir(exist_ok=True)
for f in ['main_results_unified.csv','all_families.csv','estimator_reconciliation.csv']:
    shutil.copy2(MAT/'tables'/f,OUT/'tables'/f)
shutil.copy2(MAT/'literature/references.bib',OUT/'references.bib')

REFS=[
 ('RTE France','n.d.','grid2op',r'RTE France. \emph{Grid2Op: A Testbed for Sequential Decision Making in Power Systems}. Official software, installed version 1.12.5; accessed October 13, 2026. \url{https://github.com/rte-france/Grid2Op}.'),
 ('Lajavaness','2023','ljn2023agent',r'Lajavaness. 2023. \emph{L2RPN 2023 LJN Agent}. Official source implementation. \url{https://github.com/lajavaness/l2rpn-2023-ljn-agent}.'),
 ('Hassouna et al.','2025','hassouna2025softgnn',r'Hassouna, M.; Holzh\"uter, C.; Lehna, M.; de Jong, M.; Viebahn, J.; Sick, B.; and Scholz, C. 2025. Learning Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach. In \emph{Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track and Demo Track}, 16022:129--146. doi:10.1007/978-3-032-06129-4\_8.'),
 ('de Jong, Viebahn, and Shapovalova','2025','dejong2025generalizablev1',r'de Jong, M.; Viebahn, J.; and Shapovalova, Y. 2025. Generalizable Graph Neural Networks for Robust Power Grid Topology Control. arXiv:2501.07186v1.'),
 ('Bhatia et al.','2022','bhatia2022metareasoning',r'Bhatia, A.; Svegliato, J.; Nashed, S. B.; and Zilberstein, S. 2022. Tuning the Hyperparameters of Anytime Planning: A Metareasoning Approach with Deep Reinforcement Learning. \emph{Proceedings of ICAPS}, 32:556--564. doi:10.1609/icaps.v32i1.19842.'),
 ('Milli, Lieder, and Griffiths','2017','milli2017metareasoning',r'Milli, S.; Lieder, F.; and Griffiths, T. 2017. When Does Bounded-Optimal Metareasoning Favor Few Cognitive Systems? \emph{Proceedings of AAAI}, 31. doi:10.1609/aaai.v31i1.11156.'),
]

def escape(s):
    s=s.replace('\\',r'\textbackslash{}')
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#'),('$',r'\$')]: s=s.replace(a,b)
    s=s.replace('->',r'$\rightarrow$').replace('>=',r'$\geq$').replace('<=',r'$\leq$')
    return s
def inline(s):
    s=re.sub(r'\*\*(.*?)\*\*',lambda m:'@@B'+m.group(1)+'@@E',s)
    s=escape(s).replace('@@B',r'\textbf{').replace('@@E','}')
    return re.sub(r'\[([1-6](?:,[1-6])*)\]',lambda m:r'\citep{'+','.join(REFS[int(i)-1][2] for i in m.group(1).split(','))+'}',s)

body=[];lines=source.splitlines();i=0;table_num=0
while i<len(lines):
    line=lines[i]
    if line.startswith('# '): i+=1;continue
    if line=='## References': break
    if line.startswith('## Abstract'):
        i+=1;parts=[]
        while i<len(lines) and not lines[i].startswith('## '):
            parts.append(lines[i]);i+=1
        body.append('\\begin{abstract}\n'+inline(' '.join(p for p in parts if p))+'\n\\end{abstract}')
        continue
    if line.startswith('## '):
        title=re.sub(r'^\d+\. ','',line[3:]); body.append(r'\section{'+escape(title)+'}');i+=1;continue
    if line.startswith('### '):
        title=re.sub(r'^\d+\.\d+ ','',line[4:]);body.append(r'\subsection{'+escape(title)+'}');i+=1;continue
    if line==r'\[':
        eq=[];i+=1
        while lines[i]!=r'\]': eq.append(lines[i]);i+=1
        body.append('\\begin{equation}\n'+'\n'.join(eq)+'\n\\end{equation}');i+=1;continue
    if line.startswith('```'):
        i+=1;block=[]
        while not lines[i].startswith('```'):block.append(lines[i]);i+=1
        alg=r'''\begin{algorithm}[t]
\caption{Progressive Search with Substation Completion (FRAME)}
\label{alg:frame}
\begin{algorithmic}[1]
\REQUIRE Ordered representatives $A$, query interface $Q$, aliases and $s$
\STATE $C\gets Q(\text{first }\min(128,|A|)\text{ representatives})$
\STATE $a\gets$ best admissible response in $C$
\IF{$C$ exhausts $A$, or $a$ exists and $\rho(a)<0.9$}
  \RETURN $a$
\ENDIF
\STATE $stale\gets 0$
\WHILE{unqueried ordered candidates remain}
  \STATE Query next batch of at most 64; update $C$ and $a$
  \STATE Reset $stale$ if key improved; otherwise increment it
  \IF{$a$ exists and ($\rho(a)<0.9$ or ($|C|\geq256$ and $stale\geq2$))}
    \STATE Break
  \ENDIF
\ENDWHILE
\IF{$a$ exists and $C$ does not exhaust $A$}
  \STATE $s_C\gets s(i(a))$
  \STATE $L\gets\{\operatorname{rep}(i)\in A\setminus C:s(i)=s_C\}$
  \STATE Query $L$ once; update $a$ using the same key
\ENDIF
\RETURN $a$, or no topology proposal if none exists
\end{algorithmic}
\end{algorithm}'''
        body.append(alg);i+=1;continue
    if line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].startswith('|'):
            table.append([c.strip() for c in lines[i].strip('|').split('|')]);i+=1
        table=[r for r in table if not all(re.fullmatch(r'[-:]+',c) for c in r)]
        table_num+=1
        captions={1:'Shared-controller policy comparison. FIRST retains the best admissible response if its threshold is never met.',2:'Date-level results. Positive time/query values are reductions; positive cost is worse. Confirmation query means use two nonzero-search dates.',3:'All frozen confirmation dates for FRAME; common failure and the zero-search date remain visible.'}
        n=len(table[0]);env='table*' if table_num==2 else 'table'
        if table_num==1:
            spec=r'@{}p{.25\columnwidth}p{.69\columnwidth}@{}'
        elif table_num==2:
            spec=r'@{}llrrrrr@{}'
            table[0]=['Set','Policy','Quality','Complete','Time','Queries','Cost']
            for r in table[1:]:r[1]=r[1].replace('FIRST_SAFE_BEST','FIRST')
        else:
            spec=r'@{}lrrrr@{}'
            table[0]=['Date','Steps','Cost','Time','Queries']
            for r in table[1:]:
                r[0]=r[0].replace('April 9','Apr.').replace('May 14','May').replace('November 19','Nov.')
                r[1]=r[1].split(' / ')[0];r[2]=r[2].replace('not comparable','--')
        tb=['\\begin{'+env+'}[t]',r'\centering',r'\begin{tabular}{'+spec+'}',r'\toprule']
        for j,r in enumerate(table):
            tb.append(' & '.join(inline(c) for c in r)+r' \\')
            if j==0:tb.append(r'\midrule')
        cap=r'\caption*{' if table_num==1 else r'\caption{'
        tb += [r'\bottomrule',r'\end{tabular}',cap+captions[table_num]+'}',r'\end{'+env+'}']
        body.append('\n'.join(tb));continue
    if line:body.append(inline(line)+'\n')
    else:body.append('')
    i+=1

diagram=r'''\begin{figure}[t]
\centering
\begin{tikzpicture}[node distance=6mm, every node/.style={align=center,font=\small}, box/.style={draw,rounded corners,inner sep=4pt,text width=7.1cm}]
\node[box] (state) {Current public state + base continuous action\\Legal mask, physical-action canonicalization, aliases};
\node[box,below=of state] (prefix) {Query first 128 ordered representatives\\Initial safe incumbent or exhaustion: return};
\node[box,below=of prefix] (expand) {Otherwise query batches of 64\\Stop below $\rho=0.9$, or $\geq256$ queries + two stale batches};
\node[box,below=of expand] (complete) {If extended stop + incumbent + unqueried pool:\\Query remaining aliases at original incumbent's substation};
\node[box,below=of complete] (return) {Keep best admissible observed key\\Return topology proposal to unchanged controller};
\draw[->] (state)--(prefix);\draw[->] (prefix)--(expand);\draw[->] (expand)--(complete);\draw[->] (complete)--(return);
\end{tikzpicture}
\caption{Search interface and conditional completion. The initial safe-prefix return bypasses completion; a replacement action does not trigger recursive completion.}
\label{fig:framework}
\end{figure}'''
body.insert(next(i for i,s in enumerate(body) if s.startswith(r'\section{Experimental Design}')),diagram)
header=r'''% AI-assisted author-editing draft. Not a submitted or policy-certified paper.
% Official unmodified AAAI 2027 style is used provisionally; ICAPS-specific updates take precedence.
\documentclass[letterpaper]{article}
\usepackage[preprint]{aaai2027}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\urlstyle{rm}
\def\UrlFont{\rm}
\usepackage{natbib}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{positioning}
\frenchspacing
\pdfinfo{/TemplateVersion (2027.1)}
\setcounter{secnumdepth}{1}
\title{Progressive Topology Search with Substation Completion\\for Power Grid Control}
\author{Anonymous Working Draft}
\affiliations{}
\begin{document}
\maketitle
'''
manual_bib='\n\\begin{thebibliography}{6}\n'+'\n\n'.join(r'\bibitem['+a+'('+y+')]{'+k+'}\n'+s for a,y,k,s in sorted(REFS,key=lambda r:r[0].lower()))+'\n\\end{thebibliography}\n'
tex=header+'\n'.join(body)+manual_bib+'\n\\end{document}\n'
text('manuscript.tex',tex)
sty=(KIT/'aaai2027.sty').read_text(encoding='utf-8')
standalone='\\begin{filecontents*}{aaai2027.sty}\n'+sty.rstrip()+'\n\\end{filecontents*}\n'+tex
text('manuscript_standalone.tex',standalone)
for f in ['aaai2027.sty','aaai2027.bst']:
    shutil.copy2(KIT/f,OUT/f)
dump('source_receipts.json',dict(previous_files=original_hashes,writing_repository_commit=json.loads((HERE/'reference_cache/writing_repo_commit.json').read_text())['sha'],official_kit=json.loads((HERE/'reference_cache/retrieval.json').read_text())[-1],style_unmodified=sha(OUT/'aaai2027.sty')==sha(KIT/'aaai2027.sty')))
for p in originals: assert sha(p)==original_hashes[str(p.relative_to(ROOT))]
dump('numeric_check.json',dict(passed=True,main_rows=8,family_rows=28,percent_checks=checks,all_eight_macro_rows_recomputed=True,previous_artifacts_unchanged=True,new_simulations=0,new_fits=0,limitations='Text/style preparation, not scientific reevaluation or independent AC solving.'))
print(json.dumps(dict(output=str(OUT),words=len(re.findall(r'\b[\w-]+\b',source)),numeric_checks=True,tex_generated=True),ensure_ascii=False))
