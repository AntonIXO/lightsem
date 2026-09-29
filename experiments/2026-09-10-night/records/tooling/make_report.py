"""Build a source-grounded research report from the actual audited campaign artifacts."""
from pathlib import Path
import argparse,html,json,re,sys
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest
OUT=ROOT/'delivery/research'

def read(path):return read_authenticated(ROOT/path)
def f(x,n=3):return 'n/a' if x is None else f'{x:.{n}f}'
def pct(x,n=1):return 'n/a' if x is None else f'{100*x:.{n}f}%'
def ci(s,percent=False):
    render=pct if percent else f
    return render(s['mean'])+' ['+render(s['ci95'][0])+', '+render(s['ci95'][1])+']'

class Book:
    def __init__(self):
        self.pages=[];self.sources=read('research/source_ledger.json')['sources'];self.numbers={r['id']:r['number'] for r in self.sources}
    def cite(self,*ids):return ''.join('[^'+str(self.numbers[id])+']' for id in ids)
    def page(self,title):self.pages.append({'title':title,'items':[],'references':set()})
    def p(self,text):
        self.pages[-1]['items'].append(('paragraph',text))
        self.pages[-1]['references'].update(map(int,re.findall(r'\[\^(\d+)\]',text)))
    def table(self,headers,rows,weights=None):self.pages[-1]['items'].append(('table',(headers,rows,weights)))
    def figure(self,path,caption):self.pages[-1]['items'].append(('figure',(Path(path),caption)))

def diagnostic_pages(book):
    b=book;pose=read('diagnostics/supplementary_pose.json');emb=read('diagnostics/supplementary_embeddings.json');rhythm=read('rhythm/report.json');layers=read('features/muq/layer_selection.json')
    b.page('Audio representations and the evidence for transfer')
    b.p('The useful question is whether a representation supports the local motion and lighting decisions the system actually makes. Music tagging accuracy, song identity, reconstruction quality and conditional generation are different quantities. A representation may carry timbre or genre while contributing little to the timing of a gesture; a capable decoder may also fail to use information that is present. This study therefore keeps representation probes, matched audio interventions and emitted-light measurements separate.')
    b.p('MERT learns reusable music features from acoustic and musical supervision. MuQ uses Mel residual vector quantization as a self-supervised target. Both are reasonable feature extractors to test, but their published MIR results do not establish LightSem performance. The exact released MuQ MSD checkpoint used here differs from the large-data model in the paper. The additional MERT control uses the 330M release, preserving intermediate temporal features rather than repeating the earlier PCA32 design.'+b.cite('mert','muq','muqrelease'))
    b.p('MusicFM and the layer-analysis literature motivate evaluating context and layer choice on the intended task. The last layer or a mixture of every layer is not uniformly best. Intrinsic spread and effective rank can diagnose a collapse, but they do not certify useful musical information. Here, MuQ layers 3, 6, 9 and 12 were compared using canonical-song folds inside TRAIN, without VAL or TEST selection.'+b.cite('musicfm','layerwise','goodlayer'))
    b.table(['Main input','Construction','Temporal handling'],[
        ['Base','92 existing features: 64 mel + 28 scalars','40 Hz bus; explicit beat grid retained'],
        ['Fourier','Base + 294 local flux coordinates','2 s and 8 s windows, 16 frequencies, 0.5–8 Hz'],
        ['MuQ','Base + 1,024 features from selected layer 9','Native 25 Hz; explicit interpolation to 40 Hz']], [.16,.47,.37])
    b.p('The four TRAIN layer scores were '+', '.join('layer '+str(r['layer'])+': '+f(r['score'],4) for r in layers['rows'])+'. Layer 9 follows the declared selection rule; every score was negative. This is a choice among tested layers, not evidence that the selected layer successfully predicts motion. Extraction used FP32, pinned weights, 30-second contexts and explicit audio offsets. Interpolation does not create timing information beyond the encoder’s native sampling rate.')

    b.page('Other encoders and motion-specific learning')
    b.p('BEATs is a general acoustic representation model, despite a name that resembles beat tracking. CLAP aligns audio with text and is useful evidence for semantic representation learning. OpenL3 can produce local sliding-window embeddings; treating it as necessarily a single global vector would be incorrect. Its window aggregation still matters when the downstream question concerns precise event timing. These families were reviewed but were not added to the six-arm main matrix.'+b.cite('beats','clap','openl3'))
    b.p('MotionBeat is particularly relevant as a mechanism proposal: it introduces tempo-aware and beat-jitter negatives, motion-contact guidance and bar structure. Its January 2026 version uses one pooled token per beat and 3D motion inputs. A usable released checkpoint was not verified on the project page. Consequently it is a research direction, not an executed comparison in this campaign. Any future adaptation also needs to account for alternate valid choreographies instead of automatically treating every different motion as a negative.'+b.cite('motionbeat'))
    b.p('EDGE and Listen, Denoise, Action! model audio-associated movement as a distribution. This is the appropriate distinction for the current task: one soundtrack does not determine a unique choreography. A regression decoder’s conditional mean can look static even when the motion distribution is broad. Conversely, diverse generated motion can be independent of its music. The ensemble trajectory score and matched-noise audio controls address different parts of this ambiguity.'+b.cite('edge','lda'))
    b.p('Repeated FiLM is tested as an architectural intervention, with a 32-channel bottleneck in each of six temporal-flow blocks. It begins as an identity modulation. The published FiLM mechanism supports trying feature-wise conditioning; it does not establish that this implementation cures noise dominance. Additional parameter counts are disclosed, and the study does not claim a perfectly size-matched causal comparison.'+b.cite('film'))
    b.p('No contrastive objective, new pose target or encoder fine-tuning was quietly added to the main comparison. The three input families share the original A4/B3/C17 targets, losses and scheduled sample exposure. The bounded MERT and complex-phase probes are diagnostics on TRAIN, not additional candidates that can replace an unsuccessful main arm.')

    b.page('Local periodicity and annotated rhythm')
    b.p('The Fourier block makes existing envelope periodicity and phase easier for the model to access. It is a deterministic transformation of the existing low-, mid- and high-frequency flux streams. Magnitude, real and imaginary coordinates are retained with stable normalization and zero handling. Autocorrelation and octave-folded summaries were examined separately on eight fixed TRAIN excerpts. Octave folding loses absolute tempo octave, while autocorrelation has subharmonic ambiguities.'+b.cite('fouriertutorial','cyclic','plp'))
    b.p('This envelope-phase construction should be distinguished from complex-STFT novelty. The latter uses deviations of waveform spectral phase as well as magnitude, and can therefore expose information missing from a magnitude-only cache. Harmonic/percussive structure is another useful decomposition, but it is not an oracle source separation. The exploratory complex-phase control was kept separate from the main Fourier intervention.'+b.cite('complex','hpss'))
    b.figure(OUT/'rhythm_benchmark.png','Figure 1. Real annotated rhythm benchmark. First five seconds are trimmed; F1 uses the mir_eval default 70 ms tolerance. Beat This!, when shown, is a published pretrained reference with unverified GuitarSet training overlap.')
    b.p('On 358 GuitarSet microphone excerpts, the constant grid achieved F1 '+f(rhythm['means']['constant_grid']['f1'],4)+' and CMLt '+f(rhythm['means']['constant_grid']['cmlt'],4)+'. The fixed PLP recipe achieved '+f(rhythm['means']['plp']['f1'],4)+' and '+f(rhythm['means']['plp']['cmlt'],4)+'. The paired lead-sheet F1 change was '+ci(rhythm['f1_delta'])+'; the continuity change was '+ci(rhythm['cmlt_delta'])+'. It failed both the +0.05 F1 bar and the allowed −0.02 continuity bound.'+b.cite('guitarset','guitarsettiming','beatthis'))
    b.p('All 360 original WAV files and 360 original JAMS files were checked against original per-file MD5 values. Two recordings with published timing problems were excluded before scoring. The inferential unit is the 30 lead sheets; performer, style, version, progression and downbeat summaries are supplementary. This guitar benchmark is not an EDM generalization test, and detected dance footfalls are not substituted for independent beat annotations.')

    b.page('Rhythm failure cases and descriptor reconstruction')
    b.figure(OUT/'rhythm_stress.png','Figure 2. Synthetic engineering cases. Successful tracking under tempo drift does not establish robustness to syncopation. Beat F1 alone does not test meter recognition.')
    b.p('The synthetic suite contains steady tempo, gradual drift, a tempo step, syncopation, swing, a silence break and a meter change. PLP recovers the synthetic drift better than the constant grid, but fails the syncopated example. Both methods score well on beat timing in the meter-change case because beat F1 does not ask whether the bar grouping is correct. These successes and failures remain separate from the real-data verdict.')
    b.p('The existing 24 outputs consist of 21 continuous motion descriptors and three event probabilities. They are not 24 independent coordinates of a dancer. Native symmetry is defined from side-activity balance, section energy is a smoothed motion quantity, and metric level depends on the audio beat period. Several descriptors deliberately discard pose geometry. Their semantics can be useful for lighting without making them an invertible dance representation.')
    b.table(['Probe','Seed 0 joint error','Seed 1 joint error'],[[name,f(pose['summary'][name+'/seed_0']['full']['joint_error_tl']['mean'],4),f(pose['summary'][name+'/seed_1']['full']['joint_error_tl']['mean'],4)] for name in ('descriptor128','descriptor256','pose_code24','audio_base','audio_fourier','audio_muq')],[.50,.25,.25])
    b.p('Each probe ran 12,000 scheduled updates. Targets are detected COCO-17 2D positions normalized by torso length, with low-confidence joints masked and mean horizontal root position removed. Position, velocity, root and bone-length errors are retained. The pose-conditioned 24-dimensional code is an oracle reconstruction control; it receives pose as input. It is not an audio-only system and is not deployed as a dancer in the lighting path.')

    b.page('Reconstruction, natural events and alternate coaches')
    b.figure(OUT/'pose_reconstruction.png','Figure 3. Canonical-song bootstrap intervals are shown separately for the two training seeds. The dashed line is the TRAIN mean-pose control. A low pose-conditioned reconstruction error is not audio-conditioned recovery.')
    improvements=pose['relative_pose_code_improvement_over_descriptor256']
    b.p('Relative to the wider descriptor decoder, the learned pose code reduced per-song joint error by '+ci(improvements['0'],True)+' in seed 0 and '+ci(improvements['1'],True)+' in seed 1. Increasing descriptor-decoder width from 128 to 256 produced a much smaller change. Within these tested decoder budgets, the hand-designed representation is the larger reconstruction bottleneck. This does not establish a universal information limit or prove that the learned code is a better lighting target.')
    b.p('The direct audio probes remain substantially worse than pose-conditioned reconstruction. Saved incumbent audio-to-descriptor outputs were also decoded to pose. Supplementary controls compare full outputs with a ten-second input shift, a closest-tempo foreign sequence, natural movement-event contexts, audio-onset contexts and target-past persistence. The persistence baseline has access to past pose and is unavailable to an audio-only system.')
    quality=read('diagnostics/supplementary_quality_notes.json')
    b.p('The matched foreign-prediction comparison does not establish a local advantage for any direct audio probe: each paired donor-minus-correct error interval crosses zero. Audio-onset neighborhoods cover a median '+pct(quality['coverage']['audio_onset_context']['median_fraction'])+' of frames, leaving as few as '+str(quality['coverage']['outside_audio_onset_context']['minimum_frames'])+' frames outside them in one clip. This nearly saturated segmentation is retained with its coverage and is not counted as positive event-specific evidence.')
    coaches=read('diagnostics/supplementary_coaches.json')
    b.p('The same-recording comparison contains '+str(len(coaches['rows']))+' coach pairs from '+str(coaches['summary']['pose_difference_tl']['canonical_songs'])+' canonical songs. Their mean detected-pose difference is '+ci(coaches['summary']['pose_difference_tl'])+' torso lengths. This combines choreography, body, viewing and pose-detection differences; it is not an irreducible-error estimate. Related coaches are not counted as independent songs in the reported uncertainty intervals.')

    b.page('Temporal averaging and bounded embedding probes')
    b.figure(OUT/'pose_timescales.png','Figure 4. Supplementary smoothing of saved pose predictions and targets. The nominal 0.1/0.5/2/8 s windows use 5/21/81/321 centered samples at 40 Hz; at least 80% confidence support is required.')
    b.figure(OUT/'embedding_controls.png','Figure 5. Five motion targets on 80 fixed TRAIN songs, with canonical-song folds. The constant baseline and scaling use training folds only. Intervals are 95% canonical-song bootstrap intervals for the mean, not estimates of architecture seed variance.')
    b.p('The bounded embedding probe uses a fixed ridge recipe on one-second motion summaries. Its standardized MSE values are '+', '.join(name+' '+f(emb['results'][name]['summary']['standardized_mse']['mean'],4) for name in ('base','fourier','muq','mert','complex_phase','width_control'))+', versus '+f(emb['results']['base']['summary']['train_mean_standardized_mse']['mean'],4)+' for the training-fold mean. No tested feature set improves that point estimate.')
    b.p('The original per-song R² values, including their large negative values, are preserved in the supporting data. Tiny within-song target variances make that scale easy to misread. Centered errors and training-fold standardized errors were therefore added as explicitly supplementary interpretations. These 80-song, single-coach, 60-second probes differ from the full-TRAIN layer-selection probe in sampling, regularization and coach coverage; their R² values are not interchangeable.')

def methods_page(book):
    b=book;p=read('protocol.json');pre=read('engineering_preflight.json')
    b.page('Frozen main comparison and collapse checks')
    b.p('The main matrix crosses Base, Fourier and MuQ inputs with single-entry conditioning and repeated audio FiLM, using training seeds 0 and 1. Shared modules start from matched weights. Canonical songs, recordings and coaches are sampled hierarchically. TRAIN contains 419 canonical songs and 798 coach sequences; development VAL contains 40 songs and 85 coaches. The main evaluation uses the same fixed 40 song excerpts and eight fixed 20-second rendering clips for every cell. The previously used Stage 11 TEST is not accessed.')
    b.table(['Setting','Fixed value'],[
        ['Final checkpoint','15,200 optimizer updates; no best-VAL selection'],['Training crop','32 sequences × 640 frames (16 s), 40 Hz'],
        ['Optimizer','AdamW, learning rate 3e−4, weight decay 0.01'],['Schedule','500-update warmup, common cosine schedule'],
        ['Precision','FP32; TF32 disabled'],['Outputs','A4 regression, B3 event probabilities, joint C17 temporal flow'],
        ['Generation','Six fixed noise draws; eight integration steps; 640-frame chunks'],['Selection','All mandatory gates in both seeds, then lowest mean native energy score']], [.29,.71])
    b.p('The common update budget was chosen from measured engineering step costs before main training or validation outcomes. The original 15,000-update estimate became 15,200 updates while preserving the overall approximate ten-hour allocation. Additional FiLM modules and auxiliary projections change parameter count; all six measured parameter counts are retained with the final run table. Every cell uses its final scheduled weights.')
    b.p('The primary native score is an unbiased six-draw energy score on joint C17 trajectories in nonoverlapping two-second windows. It is paired with correct-audio versus donor and within-song-shift tests, alternate noise, zero conditioning, six-feature beat-only conditioning, full donors, clock-held donors and separate core/auxiliary swaps. Native temporal variance, draw variance, effective rank and constant channels are checked separately from the actual lit-ray and encoded-video measurements.')
    b.p('A clock-held hybrid preserves six explicit rhythm inputs, including pulse clarity. It does not preserve all rhythmic information carried by MuQ or the spectral envelopes. These are constructed input interventions, not clean rhythm-free waveform counterfactuals. The A4 head remains trained and evaluated, but its four continuous outputs are not directly consumed by the new adapter; the report does not attribute emitted beam motion to those unused outputs.')

def main_pages(book):
    b=book;main=read('evaluation/report.json');runs=read('training_report.json');outcomes=read('delivery/scientific_outcomes.json');closure=read('delivery/vibescience_close.json')
    b.page('Main results and mandatory acceptance')
    cells=sorted(main['cells'],key=lambda r:(['base_once','base_film','fourier_once','fourier_film','muq_once','muq_film'].index(r['arm']),r['seed']))
    b.table(['Arm','Seed','ES ↓','Audio fraction','A donor','A local','B AP'],[[c['arm'],str(c['seed']),f(c['mean_energy_score']),f(c['controlled_song_fraction']),f(c['alignment']['mean']),f(c['local_alignment']['mean']),f(c['event_ap_advantage']['mean'])] for c in cells],[.25,.07,.10,.14,.14,.14,.16])
    b.p('ES is the joint C trajectory energy score. Audio fraction counts songs whose clock-held donor response reaches at least 15% of their spatial span and at least half their changed-noise response. A donor and A local are relative alignment advantages; their required means are 0.05 and 0.01 with positive bootstrap lower bounds. B AP requires a 0.01 advantage over beat-only with a positive lower bound. The complete per-song, per-mode values accompany the report.')
    b.figure(OUT/'gate_matrix.png','Figure 6. All mandatory checks for all 12 cells. Time σ and draw σ denote native temporal and ensemble variability; Identity denotes input/checkpoint provenance. A favorable subset does not constitute a passed configuration.')

    b.page('Feature and conditioning effects')
    b.figure(OUT/'main_comparison.png','Figure 7. Paired training seeds are shown separately. The dashed audio-response line is the absolute 0.75 acceptance threshold.')
    rows=[]
    for name in ('fourier','muq'):
        for r in outcomes['studies'][name]['details']['comparisons']:rows.append([name,r['architecture'],str(r['seed']),pct(r['relative_improvement_of_means']),ci(r['relative_improvement_macro_song'],True)])
    b.table(['Feature','Architecture','Seed','ES gain','Songwise gain [95% CI]'],rows,[.13,.18,.08,.15,.46])
    b.p('Each feature comparison uses the matched Base architecture and training seed. Both the gain of the aggregate score and the mean paired songwise gain are retained; they are different estimands. The study’s 5% aggregate improvement bar must be met without hiding an architecture or seed regression. Musical correspondence, event selectivity, native variation, brightness and visibility remain independent requirements.')
    film=outcomes['studies']['conditioning']['details']['comparisons']
    b.p('The FiLM changes in controlled-song fraction are '+', '.join(r['family']+'/seed '+str(r['seed'])+': '+f(r['change']) for r in film)+'. Repeated injection changes both conditioning structure and parameter count. These measured changes support only the tested recipe and data; a large swap response can still be wrong for the music.')

    b.page('Local MuQ use and training behavior')
    local=read('diagnostics/muq_local_controls/report.json')
    b.table(['MuQ cell','Centered vs mean-only','Centered vs centered donor'],[[r['arm']+'/seed '+str(r['seed']),ci(r['centered_energy_advantages']['aux_mean_only'],True),ci(r['centered_energy_advantages']['aux_centered_donor'],True)] for r in local['cells']],[.28,.36,.36])
    b.p('These supplementary controls remove the excerpt mean from each normalized MuQ channel, preserve only that mean, or substitute a centered donor embedding. The correct Base92 audio remains unchanged, and every control uses the same six noise draws. Both MuQ architectures and both seeds are evaluated. The local-use interpretation rule was fixed before any main candidate inference; it does not change the frozen main selection rule.')
    b.p('Centering is a distribution-shifting edit, not a separately trained centered-embedding model. Failure to retain performance can leave the mechanism unresolved rather than prove that useful local information is absent. Conversely, any claimed exclusion of a mean-only explanation requires positive paired evidence from both comparisons. The native generator tests and this mechanism test are reported as separate conclusions.')
    b.figure(OUT/'training_curves.png','Figure 8. Actual logged training losses, with the final scheduled checkpoint in every run. Loss decline is an execution and optimization observation, not a substitute for held-out conditioning or rendered-light quality.')

    b.page('Brightness, mirror pairs and independent groups')
    comp=outcomes['studies']['composition'];inc=read('evaluation/incumbent_adapter/report.json');demo=read('delivery/shpongle/observation_report.json')
    b.p('The adapter organizes six heads into three mirror pairs. A fixed policy mixes mirrored world targets with independent conditional trajectories according to predicted side-activity balance. The automatic mix is not a learned probability of geometric body symmetry. Dimmers use predicted activity, laterality and event evidence, with no fixed positive floor. Static role gains are fixed on TRAIN: wash 0.60, blinders 0.25, strobe 0.25 and projector 0.70.')
    b.figure(OUT/'adapter_comparison.png','Figure 9. Same incumbent checkpoint, previous versus new adapter, on the eight fixed VAL clips. Independent-group correlation is checked in a forced-independent diagnostic mode; the delivered film uses automatic composition.')
    b.p('The primary expressive-clip fraction changed from '+pct(comp['before'])+' to '+pct(comp['after'])+'. Adapter gates: '+', '.join(k+' '+('PASS' if v else 'FAIL') for k,v in comp['gates'].items())+'. A qualifying brightness clip needs at least four heads with encoded dimmer p95−p05 ≥0.4 and at least two heads at zero for six consecutive 40 Hz frames. Correlation is measured on emitted angular velocities, excluding intended mirror partners.')
    b.p('The mirror check compares both intended world targets and decoded emitted rays. Visibility is measured in head-only videos, so wash or projector light cannot impersonate beam motion. The same-weight comparison separates adapter effects from model retraining. It does not establish that multiple random trajectories are musically appropriate, nor does it supply fixture-level lighting-design labels.')
    b.p('Pan and tilt speed caps remain 180°/s and 120°/s. Acceleration constraints and additional fixture types were explicitly deferred. The experiment uses the canonical six-head, fourteen-fixture rig. These simulated checks do not assert real actuator feasibility or human aesthetic superiority.')

    b.page('Full-track glare and the delivered Shpongle')
    demo=read('delivery/shpongle/observation_report.json');movies=read('delivery/shpongle/render_report.json');comp=outcomes['studies']['composition']
    b.figure(ROOT/'delivery/shpongle/whole_track_luma.png','Figure 10. The entire track is measured under the same camera, exposure and shader settings. The shaded interval is the final minute; no ending-time fade or activity quota is added.')
    b.p('The delivered version reduces final-minute border luminance by '+pct(demo['border_luma_reduction'])+'; the fixed required reduction is 30%. Its glare gate is '+('PASS' if demo['ending_glare_gate'] else 'FAIL')+'. The isolated same-incumbent adapter comparison gives '+pct(comp['details']['ending_border_luma_reduction'])+'. Head-only visible motion occupies '+pct(demo['heads_only']['summary']['visible_motion_frame_fraction'])+' of observed frame pairs. Full-track thirty-second summaries and unsmoothed sampled values are included in the supporting data.')
    b.table(['Isolated ending role','Mean RGB luminance','Border luminance'],[[name,f(r['screen']['mean_luma'],5),f(r['screen']['border_luma_mean'],5)] for name,r in movies['ending_roles'].items()],[.40,.30,.30])
    b.p('Role-isolated ending renders provide a diagnostic after the static TRAIN calibration. Their light contributions are not additive because the renderer normalizes aggregate illumination. The report therefore does not sum isolated brightness values to attribute a percentage of the full scene to each role. Whole-track comparisons reveal whether a lower final-minute value merely moves the problem into an earlier interval.')
    b.p('The full new film, the previous version under the same renderer, the head-only film and a full-duration side-by-side comparison are provided. Original full-stage movies are 1920×1080 AV1; the two-panel comparison is 1920×540. Three fixed pose movies show eight reconstruction/prediction views using the correct excerpt audio offset and a common orthographic scale.')

    b.page('Computed verdicts, provenance and next decisions')
    ledger=read('delivery/compute_ledger.json');verdict={r['study']:r for r in closure['rows']}
    b.table(['Study','Computed verdict','All study gates'],[[name,verdict[name]['computed_verdict'],'PASS' if r['all_gates_pass'] else 'FAIL'] for name,r in outcomes['studies'].items()],[.28,.40,.32])
    b.p('Verdicts are computed by the experiment store from recorded before/after measurements and gate outcomes. They are not hand-assigned success labels. Fourteen component diagnostics retain the individual architecture/seed effects alongside the six fixed primary aggregates. All six hypotheses have primary-source links, interpretations and updated calibration/causal-map records.')
    b.p('The main optimizer work totals '+f(ledger['main_optimizer_seconds']/3600,3)+' hours. Recorded elapsed time of valid GPU stages totals '+f(ledger['gpu_stage_elapsed_seconds']/3600,3)+' hours. That second number includes required transfers, loading, serialization and observation inside GPU stages; it is not CUDA kernel-only utilization. Idle waiting and standalone CPU stages are excluded. The detailed ledger prevents shared training from being counted repeatedly across the six hypotheses.')
    b.p('Two technical incidents are preserved. The first addressed mel columns rather than the named flux columns; it was corrected before main training, and the incomplete affected probe was excluded. Four completed unaffected probes were retained by exact input and executable-definition audit. The second was an HF/Torch weight-normalization naming mismatch in the additional MERT control; unscored features were archived and rebuilt after strict tensor verification. MuQ passed its own complete weight audit.')
    b.p('The final audit checks all scheduled checkpoints and optimizer steps, exact target/confidence joins, fixed data and source hashes, independently recomputed trajectory scores, encoded dimmer bytes, actual video frames, mirror rays and candidate selection. A good numerical result remains bounded by development reuse, two seeds, detected 2D targets and unmeasured human preference. Those limitations are not repaired by a favorable plot or by selecting a single attractive Shpongle segment.')
    b.p('The reconstruction result justifies investigating a richer motion representation while separately testing whether audio can condition it. Rhythm changes require annotated cross-genre evidence, not synthetic success alone. Conditioning changes should be judged on correct local musical correspondence as well as response magnitude. The deferred mechanics and fixture-scaling work remain separate engineering tasks.')

def overview_page(book,draft):
    b=book;b.page('LightSem: Representation and Expressive Lighting')
    if draft:
        b.p('Main training is ongoing. This draft contains the completed diagnostic studies and the frozen methods; it does not contain final main-model, lighting-delivery or acceptance results.')
        return
    main=read('evaluation/report.json');comp=read('delivery/scientific_outcomes.json')['studies']['composition'];ledger=read('delivery/compute_ledger.json')
    b.p(('The main comparison selected '+main['chosen_arm']+' under the frozen two-seed acceptance rule.' if main['chosen_arm'] else 'None of the six new configurations passed every mandatory check in both training seeds. The delivered Shpongle follows the declared fallback: the fixed incumbent network with the new expression adapter.')+' The adapter itself is evaluated separately; its failed gates are '+(', '.join(k for k,v in comp['gates'].items() if not v) or 'none')+'.')
    b.p('The campaign completed twelve main trainings and twelve reconstruction probes on real motion data, compared a fixed local-pulse recipe on 358 independently annotated GuitarSet excerpts, and tested frozen music representations with explicit local and mean-only controls. It also measured the actual emitted DMX and GPU-rendered videos. Main optimizer work was '+f(ledger['main_optimizer_seconds']/3600,2)+' hours; the accompanying ledger reports all recorded GPU-stage elapsed time separately.')
    pose=read('diagnostics/supplementary_pose.json');rhythm=read('rhythm/report.json')
    b.p('The learned 24-dimensional pose-conditioned code reconstructs detected 2D geometry far better than the existing descriptors at both tested decoder widths. Direct audio-to-pose probes remain poor. These findings establish a gap in reconstruction under the tested models, not a claim that the learned code is already predictable from audio or ready to replace the lighting targets.')
    b.p('The local-pulse recipe scored '+f(rhythm['means']['plp']['f1'],4)+' beat F1 against '+f(rhythm['means']['constant_grid']['f1'],4)+' for the constant grid and also lost continuity. Synthetic successes under drift and swing therefore do not justify replacing the current tracker. Music-representation and architecture results are reported for every seed and architecture, with the original magnitude and absolute quality checks preserved.')
    b.p('The lighting change removes the fixed head-dimmer floor, introduces three mirror pairs with independent conditional trajectories, and applies a common TRAIN-calibrated role balance. The supplied full-track movies and before/after comparison expose the practical result. Human aesthetic preference, acceleration limits and new fixture types are outside the measured acceptance scope.')
    b.table(['Deliverable','Location'],[
        ['Full Shpongle','../shpongle/expression_v4.mp4'],['Full comparison','../shpongle/before_after.mp4'],['Heads only','../shpongle/heads_only.mp4'],
        ['Pose comparisons','../pose/'],['Chart data','chart_data.xlsx and CSV tables'],['Machine outcomes','../scientific_outcomes.json'],['Protocol','../../protocol.json']], [.32,.68])

def render(book,draft):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font=Path('/usr/share/fonts/TTF')
    for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:pdfmetrics.registerFont(TTFont(name,str(font/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
    styles={'body':ParagraphStyle('body',fontName='Body',fontSize=9.5,leading=13.4,spaceAfter=8,allowWidows=0,allowOrphans=0),
        'title':ParagraphStyle('title',fontName='Bold',fontSize=20,leading=25,spaceAfter=17,keepWithNext=True),
        'section':ParagraphStyle('section',fontName='Bold',fontSize=13.5,leading=18,spaceAfter=12,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='Body',fontSize=8,leading=10.5),
        'head':ParagraphStyle('head',fontName='Bold',fontSize=8,leading=10.5),
        'caption':ParagraphStyle('caption',fontName='Body',fontSize=8,leading=11,spaceBefore=4,spaceAfter=10),
        'footnote':ParagraphStyle('footnote',fontName='Body',fontSize=7.5,leading=10,spaceAfter=3)}
    source_by_number={r['number']:r for r in book.sources};links=[]
    def inline(text):
        text=html.escape(str(text))
        def cite(m):
            numbers=list(map(int,re.findall(r'\[\^(\d+)\]',m.group(0))));anchors=[]
            for n in numbers:
                url=source_by_number[n]['url'];links.append(url);anchors.append('<a href="'+html.escape(url,quote=True)+'">'+str(n)+'</a>')
            return '<super>'+', '.join(anchors)+'</super>'
        text=re.sub(r'(?:\[\^\d+\])+',cite,text)
        text=re.sub(r'`([^`]+)`',r'<font name="Mono" size="7.5">\1</font>',text)
        return text
    width=A4[0]-94;story=[];markdown=[]
    for pi,page in enumerate(book.pages):
        if pi:story.append(PageBreak());markdown.append('\n<!-- PAGEBREAK -->\n')
        heading=Paragraph(inline(page['title']),styles['title' if pi==0 else 'section']);heading.bookmark=page['title'];story.append(heading)
        markdown.append(('# ' if pi==0 else '## ')+page['title']+'\n')
        for kind,item in page['items']:
            if kind=='paragraph':story.append(Paragraph(inline(item),styles['body']));markdown.append(item+'\n')
            elif kind=='table':
                headers,rows,weights=item;weights=weights or [1/len(headers)]*len(headers)
                data=[[Paragraph(inline(v),styles['head' if ri==0 else 'cell']) for v in row] for ri,row in enumerate([headers]+rows)]
                table=Table(data,colWidths=[width*w/sum(weights) for w in weights],repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5e5e5')),('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                    ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7d7d7d')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d4d4d4'))]))
                story.extend([table,Spacer(1,10)]);markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
                markdown.extend('| '+' | '.join(map(str,row))+' |' for row in rows);markdown.append('')
            else:
                path,caption=item
                if not path.exists():raise FileNotFoundError(path)
                from PIL import Image as PILImage
                with PILImage.open(path) as img:ratio=img.height/img.width
                image=Image(str(path),width=width,height=width*ratio)
                story.append(KeepTogether([image,Paragraph(inline(caption),styles['caption'])]));markdown.append('!['+caption+']('+str(path)+')\n')
        if page['references']:
            story.append(Spacer(1,7))
            for number in sorted(page['references']):
                r=source_by_number[number]
                note=f"{number}. {r['authors']}; {r['publication']}. "+r['title']
                story.append(Paragraph('<a href="'+html.escape(r['url'],quote=True)+'">'+html.escape(note)+'</a>',styles['footnote']))
    story.append(PageBreak());story.append(Paragraph('Sources',styles['section']));markdown.append('\n## Sources\n')
    for r in book.sources:
        text=f"{r['number']}. {r['authors']}. {r['title']}. {r['publication']}."
        story.append(Paragraph('<a href="'+html.escape(r['url'],quote=True)+'">'+html.escape(text)+'</a>',styles['body']))
        markdown.append(f"[^{r['number']}]: {r['authors']}. [{r['title']}]({r['url']}). {r['publication']}.\n")
    name='diagnostics_draft' if draft else 'LightSem_Overnight_Study'
    pdf=OUT/(name+'.pdf');md=OUT/(name+'.md');md.write_text('\n'.join(markdown))
    class Doc(SimpleDocTemplate):
        def afterFlowable(self,flowable):
            if hasattr(flowable,'bookmark'):
                key='section-'+str(len(getattr(self,'bookmarks',[])));self.bookmarks=getattr(self,'bookmarks',[])+[key]
                self.canv.bookmarkPage(key);self.canv.addOutlineEntry(flowable.bookmark,key,0)
    doc=Doc(str(pdf),pagesize=A4,leftMargin=47,rightMargin=47,topMargin=45,bottomMargin=45,title=book.pages[0]['title'],author='LightSem',subject='Audited representation, rhythm and expressive lighting study')
    doc.build(story)
    authenticated_json(OUT/(name+'_manifest.json'),{'pdf':str(pdf),'pdf_sha256':digest(pdf),'markdown':str(md),'markdown_sha256':digest(md),
        'draft':draft,'sections':len(book.pages),'source_sha256':digest(__file__),'primary_links':sorted(set(links))})
    print(json.dumps({'pdf':str(pdf),'sections':len(book.pages),'draft':draft}))

def main(draft=False):
    OUT.mkdir(parents=True,exist_ok=True);b=Book();overview_page(b,draft);diagnostic_pages(b);methods_page(b)
    if not draft:main_pages(b)
    render(b,draft)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--draft',action='store_true');a=ap.parse_args();main(a.draft)
