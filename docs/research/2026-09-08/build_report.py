"""Render the frozen research synthesis; CPU-only ReportLab artifact tooling."""
from pathlib import Path
import re
import html
import json
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'report-source.md'
OUTPUT = ROOT / 'LightSem_Research_Review_2026-09-08.pdf'
FONT = Path('/usr/share/fonts/TTF')
for name, filename in [('Body', 'DejaVuSans.ttf'), ('BodyBold', 'DejaVuSans-Bold.ttf'), ('BodyItalic', 'DejaVuSans-Oblique.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='BodyItalic', boldItalic='BodyBold')

INK = colors.HexColor('#152C3A')
TEAL = colors.HexColor('#007F83')
MUTED = colors.HexColor('#546771')
PALE = colors.HexColor('#EEF6F5')
LINE = colors.HexColor('#D8E3E5')
PAGE_W, PAGE_H = A4
MARGIN = 47
WIDTH = PAGE_W - 2 * MARGIN

styles = {
    'body': ParagraphStyle('body', fontName='Body', fontSize=9.5, leading=13.35, textColor=INK, spaceAfter=7.4, allowWidows=0, allowOrphans=0),
    'h1': ParagraphStyle('h1', fontName='BodyBold', fontSize=21, leading=25.5, textColor=INK, spaceAfter=17, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='BodyBold', fontSize=11.4, leading=15.2, textColor=TEAL, spaceBefore=8, spaceAfter=7.5, keepWithNext=True),
    'bullet': ParagraphStyle('bullet', fontName='Body', fontSize=9.5, leading=13.35, textColor=INK, leftIndent=12, firstLineIndent=-10, spaceAfter=8),
    'quote': ParagraphStyle('quote', fontName='BodyBold', fontSize=10.3, leading=16.2, textColor=TEAL, leftIndent=13, rightIndent=13, spaceBefore=7, spaceAfter=12),
    'cell': ParagraphStyle('cell', fontName='Body', fontSize=8.15, leading=11.3, textColor=INK, spaceAfter=0),
    'cellhead': ParagraphStyle('cellhead', fontName='BodyBold', fontSize=8.3, leading=11.3, textColor=colors.white),
    'refs': ParagraphStyle('refs', fontName='Body', fontSize=8.6, leading=12, textColor=INK, spaceAfter=9, allowWidows=0, allowOrphans=0),
}
link_records = []

def inline(text):
    slots = []
    def link(match):
        label, href = match.group(1), match.group(2)
        if not href.startswith(('https://', 'http://')):
            resolved = (ROOT / href).resolve()
            if not resolved.exists():
                raise FileNotFoundError(resolved)
            href = str(resolved)
        link_records.append({'label': label, 'target': href})
        token = f'LINKTOKEN{len(slots)}END'
        slots.append(f'<a href="{html.escape(href, quote=True)}" color="#007F83">{html.escape(label)}</a>')
        return token
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
    text = html.escape(text)
    text = re.sub(r'`([^`]+)`', r'<font name="Mono" size="8.1">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    for i, replacement in enumerate(slots):
        text = text.replace(f'LINKTOKEN{i}END', replacement)
    return text

def decorate(c, doc):
    c.saveState()
    c.setFillColor(TEAL)
    c.rect(MARGIN, PAGE_H - 36, 25, 3, stroke=0, fill=1)
    c.setFont('BodyBold', 8)
    c.drawString(MARGIN + 34, PAGE_H - 36, 'LIGHTSEM  /  RESEARCH REVIEW')
    c.setFillColor(MUTED)
    c.setFont('Body', 7.5)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 36, '8 SEPTEMBER 2026')
    c.setStrokeColor(LINE)
    c.line(MARGIN, 38, PAGE_W - MARGIN, 38)
    c.setFont('Body', 7)
    c.drawString(MARGIN, 24, 'Primary sources + local code and experiment evidence')
    c.drawRightString(PAGE_W - MARGIN, 24, f'{doc.page:02d}')
    c.restoreState()

def make_table(lines):
    rows = [[c.strip() for c in row.strip().strip('|').split('|')] for row in lines]
    rows = [r for r in rows if not all(re.fullmatch(r'[- :]+', c) for c in r)]
    data = [[Paragraph(inline(c), styles['cellhead' if i == 0 else 'cell']) for c in row] for i, row in enumerate(rows)]
    table = Table(data, colWidths=[WIDTH * .29, WIDTH * .35, WIDTH * .36], repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), INK),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [PALE, colors.white]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LINEBELOW', (0, 0), (-1, -1), .35, LINE),
    ]))
    return table

sections = SOURCE.read_text().split('<!-- PAGEBREAK -->')
story = []
for section_index, section in enumerate(sections):
    if section_index:
        story.append(PageBreak())
    lines = section.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i]); i += 1
            story.extend([make_table(table_lines), Spacer(1, 10)])
            continue
        if line.startswith('# '):
            style, value = 'h1', line[2:]
        elif line.startswith('## '):
            style, value = 'h2', line[3:]
        elif line.startswith('- '):
            style, value = 'bullet', '• ' + line[2:]
        elif line.startswith('> '):
            style, value = 'quote', line[2:]
        else:
            style, value = ('refs' if section_index >= 10 else 'body'), line
        paragraph = Paragraph(inline(value), styles[style])
        if style == 'h1':
            paragraph._bookmark_title = value
        story.append(paragraph)
        i += 1

class ResearchDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        title = getattr(flowable, '_bookmark_title', None)
        if title:
            key = f'section-{self.page}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, 0)

doc = ResearchDoc(str(OUTPUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                  topMargin=64, bottomMargin=51,
                  title='LightSem: Where the Research Should Go Next',
                  author='LightSem research review', subject='Incremental literature review and proposed experiments, 2026-09-08')
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
(ROOT / 'pdf-links.json').write_text(json.dumps(link_records, indent=2))
print(json.dumps({'pdf':str(OUTPUT), 'bytes':OUTPUT.stat().st_size, 'intended_sections':len(sections), 'links':len(link_records)}))
