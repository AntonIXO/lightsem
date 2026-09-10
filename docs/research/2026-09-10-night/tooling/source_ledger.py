"""Verified primary sources for the overnight research synthesis and vault links."""
from pathlib import Path
import json,sys
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
SOURCES=[
 ('mert','Li et al.','MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training','ICLR 2024','https://arxiv.org/html/2306.00107v4',
  'Music representations from acoustic and musical self-supervision; sequence-level MIR performance does not establish fine-grained LightSem motion transfer. Local control uses the330M checkpoint,75 Hz features and layers15/16.', ['muq','fourier']),
 ('muq','Zhu et al.','MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization','2025','https://arxiv.org/pdf/2501.01108',
  'Mel-RVQ targets and layer-specific music representations motivate a frozen encoder test. The public MSD release is distinct from the large-data paper model; the study uses25 Hz native features and FP32 extraction.', ['muq']),
 ('musicfm','Won, Hung and Le','A Foundation Model for Music Informatics','ICASSP 2024','https://arxiv.org/pdf/2311.03318',
  'Token-level tasks and longer context reveal strengths not captured by global music tagging. This is a reason to test local temporal behavior, not proof of useful dance conditioning.', ['muq','rhythm']),
 ('layerwise','Zhou, Zhu and Chen','Layer-wise Investigation of Large-Scale Self-Supervised Music Representation Models','2025','https://arxiv.org/html/2505.16306v1',
  'Intermediate layers can be better suited to downstream tasks than default final-layer or all-layer aggregation. The reported tasks do not directly establish beat or LightSem motion quality.', ['muq']),
 ('goodlayer','Kanatas, Kong, Alonso-Jiménez, Serra and Bogdanov','What Makes a Good Layer? Assessing the Layer-Wise Intrinsic Properties of Music Foundation Models','ISMIR 2026','https://arxiv.org/html/2608.14819v1',
  'Across12 models and15 tasks, intrinsic geometric criteria are task-dependent. Effective rank or representation spread alone is not a universal quality certificate.', ['muq','conditioning']),
 ('beats','Chen et al.','BEATs: Audio Pre-Training with Acoustic Tokenizers','ICML 2023','https://arxiv.org/abs/2212.09058',
  'General audio representation learning with acoustic tokenizers; the name BEATs does not denote a music beat tracker.', ['muq','rhythm']),
 ('clap','Wu et al.','Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation','2023','https://arxiv.org/html/2211.06687v4',
  'Audio-text alignment supports semantic transfer but is not a direct validation of frame-local rhythm or motion control.', ['muq']),
 ('openl3','Cramer, Wu, Salamon and Bello','Look, Listen and Learn More: Design Choices for Deep Audio Embeddings','ICASSP 2019','https://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf',
  'Audio-visual pretraining yields audio representations; local sliding-window use is possible. One-second aggregation should be distinguished from precise within-window timing.', ['muq']),
 ('motionbeat','Wang, Wang and Cai','MotionBeat: Motion-Aligned Music Representation via Embodied Contrastive Learning and Bar-Equivariant Contact-Aware Encoding','ICASSP 2026; arXiv v2, 29 Jan 2026','https://arxiv.org/html/2510.13244v2',
  'Tempo-aware and beat-jitter negatives, motion contacts and bar structure motivate harder local controls. Inputs are pooled to one token per beat and rely on3D motion; released usable weights were not verified on the project page. This is a future mechanism, not an executed checkpoint comparison.', ['muq','conditioning','representation']),
 ('cyclic','Grosche, Müller and Kurth','Cyclic Tempogram—A Mid-level Tempo Representation for Music Signals','ICASSP 2010','https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf',
  'Octave-folded periodicity exposes cyclic tempo relations while losing absolute tempo octave. Keep unfolded frequency information alongside cyclic summaries.', ['fourier','rhythm']),
 ('plp','Grosche and Müller','Extracting Predominant Local Pulse Information from Music Recordings','2011; author reference notebook','https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S3_PredominantLocalPulse.html',
  'Local Fourier periodicity and phase support a time-varying pulse estimate. Benefit over a fixed grid must be checked on annotated music rather than inferred from the method definition.', ['fourier','rhythm']),
 ('complex','Duxbury, Bello, Davies and Sandler','Complex Domain Onset Detection for Musical Signals','DAFx 2003','https://dafx.de/paper-archive/2003/pdfs/dafx81.pdf',
  'Joint phase and magnitude deviations motivate phase-sensitive novelty computed from the original complex STFT. This uses information absent from an existing magnitude-only cache.', ['fourier','rhythm']),
 ('hpss','FitzGerald','Harmonic/Percussive Separation Using Median Filtering','DAFx 2010','https://www.audiolabs-erlangen.de/resources/aps-w23/papers/2010_FitzGerald_HarmonicPercussiveSep_DAFx.pdf',
  'Spectrogram structure can support harmonic/percussive decomposition. It does not provide oracle source stems or establish motion predictability.', ['fourier']),
 ('beatthis','Foscarin, Schlüter and Widmer','Beat This! Accurate Beat Tracking Without DBN Postprocessing','ISMIR 2024','https://arxiv.org/pdf/2407.21658',
  'Beat F1 and continuity are distinct evaluation axes. The released final0 checkpoint is a comparative tracker, not ground-truth annotation, and overlap with GuitarSet is unverified here.', ['rhythm']),
 ('edge','Tseng, Castellon and Liu','EDGE: Editable Dance Generation From Music','CVPR 2023','https://arxiv.org/html/2211.10658v1',
  'Music-conditioned diffusion with cross-attention and physical/diversity checks illustrates the need to evaluate several motion-quality axes. It does not validate this study\'s FiLM intervention or lighting adapter.', ['representation','conditioning','composition']),
 ('lda','Alexanderson, Nagy, Beskow and Henter','Listen, Denoise, Action! Audio-Driven Motion Synthesis with Diffusion Models','SIGGRAPH 2023','https://arxiv.org/html/2211.09707v2',
  'Audio-associated movement is ambiguous and is appropriately treated probabilistically. Conditional mean reconstruction and distributional generation answer different questions.', ['representation','conditioning','composition']),
 ('film','Perez et al.','FiLM: Visual Reasoning with a General Conditioning Layer','AAAI 2018','https://arxiv.org/pdf/1709.07871',
  'Feature-wise affine modulation is a general conditioning mechanism. Transfer to repeated audio conditioning in this temporal flow is an experimentally tested hypothesis.', ['conditioning']),
 ('guitarset','GuitarSet authors','GuitarSet v1.1.0','2019 dataset release','https://zenodo.org/records/3371780',
  'Independent microphone audio and beat-position JAMS annotations. This study uses358 of 360 performances after excluding two known timing problems, and verifies every original WAV/JAMS MD5.', ['rhythm']),
 ('guitarsettiming','GuitarSet maintainers','Annotation/audio timing issue 5','Repository issue','https://github.com/marl/GuitarSet/issues/5',
  'Published timing problems motivate the two predefined whole-file exclusions:04_BN3-154-E_comp and04_Jazz1-200-B_comp.', ['rhythm']),
 ('muqrelease','Tencent AI Lab / OpenMuQ','MuQ-large-msd-iter released model','Pinned revision 0562a57814f6f8bbd9fdea0a25921a2fce1a841a','https://huggingface.co/OpenMuQ/MuQ-large-msd-iter',
  'The exact public MSD release used for feature extraction. Checkpoint metadata has CC-BY-NC 4.0; code licensing is separate. All 525 loaded tensors matched the pinned checkpoint.', ['muq']),
 ('fouriertutorial','Müller and collaborators','Fourier Tempogram','Author FMP notebook','https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S2_TempogramFourier.html',
  'Local Fourier coefficients of novelty retain frequency and phase information. A deterministic transformation of an existing envelope changes its representation, not its Shannon information content.', ['fourier','rhythm'])
]

def write():
    sys.path.insert(0,str(ROOT/'source'))
    from lightsem.experiments.stage11_common import authenticated_json,digest
    rows=[{'number':i+1,'id':key,'authors':authors,'title':title,'publication':publication,'url':url,'claim_and_scope':claim,'hypotheses':hypotheses} for i,(key,authors,title,publication,url,claim,hypotheses) in enumerate(SOURCES)]
    authenticated_json(ROOT/'research/source_ledger.json',{'sources':rows,'source_sha256':digest(__file__),'verified_primary_sources':True})
    return rows

def link():
    rows=write();sys.path.insert(0,'/root/vibescience-mcp/src')
    from vibescience_mcp.core import Store
    store=Store('/root/lightsem-vault');links=[]
    for row in rows:
        paper_id='night-'+row['id']+'-20260909'
        if not store.vault.exists('papers',paper_id):
            store.add_paper(id=paper_id,title=row['title'],arxiv_id_or_url=row['url'],key_claims=[row['claim_and_scope']])
        for suffix in row['hypotheses']:
            target='h-night-'+suffix+'-20260909';store.link_paper(target,paper_id);links.append({'hypothesis':target,'paper':paper_id})
        store.link_paper('dancing-light-generative-20260903',paper_id)
    from lightsem.experiments.stage11_common import authenticated_json
    authenticated_json(ROOT/'research/vault_paper_links.json',{'links':links,'vault':'/root/lightsem-vault'})
    print(json.dumps({'papers':len(rows),'hypothesis_links':len(links)}))

if __name__=='__main__':
    if '--link' in sys.argv:link()
    else:write()
