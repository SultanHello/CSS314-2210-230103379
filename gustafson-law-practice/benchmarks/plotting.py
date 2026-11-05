"""Vector PDF plots using ReportLab; matching 300 dpi PNGs using PDFium."""
from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import pypdfium2

COLORS = ['#94a3b8', '#2563eb', '#dc2626', '#059669', '#9333ea']

def line_plot(path, title, subtitle, series, xlabel='Processes P', ylabel='Speedup', logx=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    width,height=720,460
    c=canvas.Canvas(str(path),pagesize=(width,height))
    left,bottom,right,top=76,80,680,355
    allpoints=[point for s in series for point in s['points']]
    xmin,xmax=min(p[0] for p in allpoints),max(p[0] for p in allpoints)
    ymin=0.0; ymax=max((p[1]+(p[2] if len(p)>2 else 0)) for p in allpoints)*1.08
    if ymax<=0: ymax=1
    def tx(x):
        a,b,v=(math.log2(xmin),math.log2(xmax),math.log2(x)) if logx else (xmin,xmax,x)
        return left+(v-a)/(b-a)*(right-left) if b>a else left
    def ty(y):return bottom+(y-ymin)/(ymax-ymin)*(top-bottom)
    c.setFillColor(HexColor('#0f172a')); c.setFont('Helvetica-Bold',18); c.drawString(left,421,title)
    c.setFont('Helvetica',9); c.drawString(left,403,subtitle)
    for i in range(6):
        y=ymax*i/5
        c.setStrokeColor(HexColor('#e2e8f0'));c.setLineWidth(.5);c.line(left,ty(y),right,ty(y))
        c.setFillColor(HexColor('#475569'));c.setFont('Helvetica',9);c.drawRightString(left-10,ty(y)-3,f'{y:.2g}')
    if logx:
        ticks=[2**i for i in range(math.ceil(math.log2(xmin)),math.floor(math.log2(xmax))+1)]
        if len(ticks)>10:ticks=ticks[::2]+([ticks[-1]] if ticks[-1] not in ticks[::2] else [])
    else:
        ticks=sorted(set(p[0] for p in allpoints))
        if len(ticks)>12:ticks=[xmin+(xmax-xmin)*i/5 for i in range(6)]
    for x in ticks:
        c.setFillColor(HexColor('#475569'));c.drawCentredString(tx(x),bottom-17,f'{x:g}')
    c.setStrokeColor(HexColor('#64748b'));c.line(left,bottom,right,bottom);c.line(left,bottom,left,top)
    c.setFont('Helvetica',10);c.drawCentredString((left+right)/2,35,xlabel+(' (log2 axis)' if logx else ''))
    c.saveState();c.translate(22,(bottom+top)/2);c.rotate(90);c.drawCentredString(0,0,ylabel);c.restoreState()
    for i,s in enumerate(series):
        color=HexColor(s.get('color',COLORS[i%len(COLORS)]));c.setStrokeColor(color);c.setFillColor(color)
        c.setLineWidth(s.get('width',1.6));c.setDash(4,3) if s.get('dash') else c.setDash()
        if s.get('connect',True):
            pathline=c.beginPath()
            for j,p in enumerate(s['points']):
                if j==0:pathline.moveTo(tx(p[0]),ty(p[1]))
                else:pathline.lineTo(tx(p[0]),ty(p[1]))
            c.drawPath(pathline)
        c.setDash()
        if s.get('markers'):
            for p in s['points']:
                x,y=tx(p[0]),ty(p[1]);c.circle(x,y,3,stroke=1,fill=1)
                if len(p)>2:
                    lo,hi=ty(p[1]-p[2]),ty(p[1]+p[2]);c.line(x,lo,x,hi);c.line(x-4,lo,x+4,lo);c.line(x-4,hi,x+4,hi)
        # Two-column legend outside the plotting region.
        lx=left+(i%2)*310;ly=383-(i//2)*15
        c.line(lx,ly,lx+22,ly);c.setFont('Helvetica',9);c.drawString(lx+29,ly-3,s['label'])
    c.showPage();c.save()
    doc=pypdfium2.PdfDocument(str(path));page=doc[0];bitmap=page.render(scale=300/72)
    bitmap.to_pil().save(path.with_suffix('.png'));bitmap.close();page.close();doc.close()
