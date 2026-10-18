"""Rebuild all vector/preview outputs with actual diagrams.net exports."""
from pathlib import Path
import subprocess, os, json, re
from pypdf import PdfReader, PdfWriter
from PIL import Image

ROOT=Path('D:/Codex/GRL研究')
OUT=ROOT/'outputs/frame_motivation_v1_20261018'
DRAW=Path('D:/Codex/tools/drawio-32.4.1/draw.io.exe')
POP=Path('C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
BASE='FRAME_动机图_V1'

def main():
    env=dict(os.environ,TEMP=str(ROOT/'tmp'),TMP=str(ROOT/'tmp'))
    for ext,target in [('svg',OUT/(BASE+'.svg')),('pdf',OUT/'intermediate'/(BASE+'.raw.pdf'))]:
        target.parent.mkdir(exist_ok=True)
        command=[str(DRAW),'--user-data-dir=D:/Codex/tools/drawio-profile','--export','--format',ext,
                 '--embed-diagram' if ext=='svg' else '--crop',
                 '--output',str(target),str(OUT/(BASE+'.drawio'))]
        subprocess.run(command,check=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        assert target.exists() and target.stat().st_size>1000
    svgpath=OUT/(BASE+'.svg')
    svg=svgpath.read_text(encoding='utf-8')
    match=re.search(r'<svg\b[^>]*>',svg)
    assert match
    tag=match.group()
    tag=re.sub(r'\bwidth="[^"]*"','width="170mm"',tag,count=1)
    tag=re.sub(r'\bheight="[^"]*"','height="112.8181818182mm"',tag,count=1)
    svgpath.write_text(svg[:match.start()]+tag+svg[match.end():],encoding='utf-8')
    p=PdfReader(OUT/'intermediate'/(BASE+'.raw.pdf')).pages[0]
    p.scale_to(170/25.4*72,170/25.4*72*730/1100)
    writer=PdfWriter();writer.add_page(p)
    writer.add_metadata({'/Title':'FRAME motivation: depth and location',
                         '/Subject':'Schematic. Electricity Pole by Georgiana Ionescu / Noun Project, CC BY 3.0 US, recolored and resized.'})
    writer.write(OUT/(BASE+'.pdf'))
    subprocess.run([str(POP),'-scale-to','2200','-singlefile','-png',str(OUT/(BASE+'.pdf')),
                    str(OUT/BASE)],check=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    im=Image.open(OUT/(BASE+'.png'))
    im.convert('L').save(OUT/'灰度检查.png')
    im.resize((670,round(im.height*670/im.width)),Image.Resampling.LANCZOS).save(OUT/'论文宽度检查.png')
    print(json.dumps({'pages':1,'raster_images_in_pdf':len(p.images),'pdf_width_mm':170,
                      'preview_px':list(im.size)},ensure_ascii=False))

if __name__=='__main__':main()
