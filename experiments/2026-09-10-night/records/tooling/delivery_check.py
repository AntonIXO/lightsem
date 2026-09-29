"""Verify completed delivery manifests and spreadsheet rows without rerunning experiments."""
from pathlib import Path
import csv,json,subprocess,sys
from openpyxl import load_workbook
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest,verify_artifacts,artifact_manifest

def main():
    research=ROOT/'delivery/research'
    for filename,flag in [('pipeline_state.json','gpu_pipeline_complete'),('post_pipeline_state.json','post_pipeline_complete')]:
        state=read_authenticated(ROOT/filename)
        assert state[flag] and not state.get('failure')
        for stage in state['stages']:verify_artifacts(stage['artifacts'])
    report=read_authenticated(research/'LightSem_Overnight_Study_manifest.json')
    assert report['draft'] is False and len(report['primary_links'])==21
    for name in ['pdf','markdown']:assert digest(report[name])==report[name+'_sha256']
    figures=read_authenticated(research/'figure_manifest.json')
    for name,sha in figures['artifacts'].items():assert digest(research/name)==sha,name
    wb=load_workbook(research/'chart_data.xlsx',read_only=True,data_only=True)
    counts={}
    for name,expected in figures['tables'].items():
        count=sum(1 for _ in wb[name[:31]].iter_rows(min_row=2,values_only=True))
        with (research/(name+'.csv')).open() as stream:csv_count=sum(1 for _ in csv.DictReader(stream))
        assert count==expected==csv_count,(name,count,expected,csv_count)
        counts[name]=count
    sources=sum(1 for row in wb['Sources'].iter_rows(min_row=2,values_only=True) if str(row[0]).startswith('Primary source '))
    assert sources==21
    wb.close()
    closed=read_authenticated(ROOT/'delivery/vibescience_close.json')
    assert len(closed['rows'])==6 and len({r['experiment_id'] for r in closed['rows']})==6
    outcomes=read_authenticated(ROOT/'delivery/scientific_outcomes.json')
    assert all(r['all_study_gates_pass']==outcomes['studies'][r['study']]['all_gates_pass'] for r in closed['rows'])
    info=subprocess.check_output(['pdfinfo',report['pdf']],text=True)
    pages=int(next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
    pdf_text=subprocess.check_output(['pdftotext','-layout',report['pdf'],'-'],text=True)
    page_text=[p.strip() for p in pdf_text.split('\f') if p.strip()]
    assert len(page_text)==pages and all(len(p)>500 for p in page_text),'Empty or orphan report page'
    # ReportLab writes URI annotations without compressing their dictionaries.
    pdf_bytes=Path(report['pdf']).read_bytes()
    assert all(url.encode() in pdf_bytes for url in report['primary_links']),'Missing PDF citation URI'
    paths=[Path(report['pdf']),Path(report['markdown']),research/'chart_data.xlsx',research/'figure_manifest.json',research/'LightSem_Overnight_Study_manifest.json',ROOT/'delivery/scientific_outcomes.json',ROOT/'delivery/vibescience_close.json',ROOT/'delivery/compute_ledger.json',ROOT/'protocol.json',ROOT/'post_pipeline_protocol.json']
    paths+=list((ROOT/'delivery/shpongle').glob('*.mp4'))+list((ROOT/'delivery/pose').glob('*.mp4'))+list((ROOT/'diagnostics').glob('independent_*audit.json'))
    result={'passed':True,'report_pages':pages,'primary_source_links':sources,'table_rows':counts,'closed_studies':{r['study']:r['computed_verdict'] for r in closed['rows']},'artifacts':artifact_manifest(paths),'source_sha256':digest(__file__),'scope':'Delivery hashes, completed stage manifests, workbook/CSV row counts, PDF page and citation checks. Scientific scores and videos were checked by the separate independent audits.'}
    authenticated_json(ROOT/'delivery/delivery_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','scope','source_sha256')}))

if __name__=='__main__':main()
