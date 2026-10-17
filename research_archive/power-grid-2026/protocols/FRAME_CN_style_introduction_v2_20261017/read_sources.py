from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, hashlib
from urllib.request import Request, urlopen
from html.parser import HTMLParser

class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'):self.skip+=1
    def handle_endtag(self,tag):
        if tag in ('script','style'):self.skip=max(0,self.skip-1)
        if tag in ('p','div','h1','h2','h3','h4','li'):self.parts.append('\n')
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)

root=Path('D:/Codex/GRL研究/outputs/frame_cn_style_20261017')
out=root/'sources'
out.mkdir(parents=True,exist_ok=True)
sources=[
 ('SJTU2021_topology','https://xuebao.sjtu.edu.cn/article/2021/1006-2467/1006-2467-55-S2-7.shtml'),
 ('CES2025_powerflow','https://dgjsxb.ces-transaction.com/fileup/HTML/2025-19-6115.htm'),
 ('CEPC2026_recovery','https://www.cepc.com.cn/CN/abstract/article/1000-7229/7263'),
]
def fetch(item):
    key,url=item
    try:
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=45) as response:
            raw=response.read();status=response.status
        (out/(key+'.html')).write_bytes(raw)
        parser=TextParser();parser.feed(raw.decode('utf-8',errors='replace'))
        text='\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())
        (out/(key+'.txt')).write_text(text,encoding='utf-8')
        return dict(key=key,url=url,status=status,sha256=hashlib.sha256(raw).hexdigest(),text_characters=len(text))
    except Exception as e:
        return dict(key=key,url=url,error=str(e))
with ThreadPoolExecutor(max_workers=3) as pool:
    receipts=list(pool.map(fetch,sources))
(root/'source_receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipts,ensure_ascii=False,indent=2))
