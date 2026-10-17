from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from html.parser import HTMLParser
import hashlib,json

class Parser(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.links=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
        if tag=='a':self.links.append(dict(attrs).get('href',''))
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.skip=max(0,self.skip-1)
        if tag in ('div','p','h1','h2','h3','h4','li'):self.parts.append('\n')
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)

root=Path('D:/Codex/GRL研究/outputs/frame_related_section_reading_20261017')
out=root/'sources';out.mkdir(parents=True,exist_ok=True)
items=[
('RLO2020','https://zhipeng93.github.io/files/rlo-sci-china2020.pdf','pdf'),
('AAS2023_knowledge','https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c220629?viewType=HTML','html'),
('AAS2024_alignment','https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c210518?viewType=HTML','html'),
('SSI2024_scheduling','https://www.sciengine.com/doi/pdf/FD7B3F44B3AE4E1AA7985C72AD15EE96','pdf'),
('JOS2025_issue','https://www.jos.org.cn/jos/article/issue/2025_36_8','html'),
]
def fetch(item):
    key,url,kind=item
    try:
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as r:raw=r.read()
        (out/(key+'.'+kind)).write_bytes(raw)
        info=dict(key=key,url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if kind=='html':
            p=Parser();p.feed(raw.decode('utf-8',errors='replace'))
            (out/(key+'.txt')).write_text('\n'.join(x.strip() for x in ''.join(p.parts).splitlines() if x.strip()),encoding='utf-8')
            if key=='JOS2025_issue':info['matching_links']=[s for s in p.links if '7354' in s]
        return info
    except Exception as e:return dict(key=key,url=url,error=str(e))
with ThreadPoolExecutor(max_workers=5) as pool:receipts=list(pool.map(fetch,items))
(root/'source_receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipts,ensure_ascii=False,indent=2))
