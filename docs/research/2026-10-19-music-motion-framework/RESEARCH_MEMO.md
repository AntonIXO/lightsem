# Music–Motion Causal Audit (MMCA) — research memo

Дата: 2026-10-19
Статус: проект протокола и CPU-реализация; не является квалификацией
foundation-модели.

## Фокус: новая foundation-модель

MMCA предназначен прежде всего для новой foundation-модели. Его главный
объект — связка `MuQ/Base92 → foundation latent → native body/light heads`.
Старый native dance pipeline используется как frozen reference для raw-body
safe limits и как downstream retention check после mapper/DMX. Он не является
target distribution для нового critic и не должен «подтягивать» плохой
foundation output процедурным движением.

Для foundation сохраняем три независимых артефакта на каждый трек:

1. latent slots/representation до output heads;
2. raw 30 Hz body или raw learned aim до mapper;
3. actual decoded DMX/emitted streams после mapper.

Так можно различить latent collapse, output-head collapse и loss of information
на body→light seam. Latent effective rank сам по себе не является musicality:
его нужно связывать с audio predictive gain и own-song counterfactuals.

## Почему текущие проверки недостаточны

Существующий `autonomous-training-evaluation` корректно ловит nonfinite raw
body, speed/acceleration/jerk, inactive outputs, exact duplicates, diversity и
простую input sensitivity. Это необходимо, но не отвечает на вопрос «движение
выбрано под эту музыку». В foundation-v2 уже наблюдался правильный по форме
провал: finite outputs и RGB/brightness response при почти тождественной
межпесенной aim trajectory (`cross-song aim corr≈0.9963`). Поэтому hash,
nonzero movement или одна корреляция с onset не принимаются как музыкальность.

## Что подтверждает литература

* **Beat Alignment Score (BAS / BeatAlign).** AI Choreographer на AIST++
  определяет kinematic beats из локальных минимумов kinetic velocity и
  сопоставляет их с ближайшими музыкальными beats через Gaussian distance.
  Это полезный первый timing-инструмент, но он однонаправленный и не проверяет
  phrase/style; метроном с событием на каждом beat может получить высокий score.
  [AI Choreographer (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_AI_Choreographer_Music_Conditioned_3D_Dance_Generation_With_AIST_ICCV_2021_paper.pdf).
* **Style consistency.** MDSC показывает, что прямое расстояние между
  независимыми audio/motion embeddings плохо определяет стиль; авторы обучают
  joint-space кластеризацию и проверяют intra/inter-cluster ratio, сверяя его с
  human study. Это хороший второй слой, но он требует отдельного frozen
  motion/audio encoder и проверки domain shift. [MDSC](https://arxiv.org/abs/2309.01340).
* **Motion-aligned audio representations.** MotionBeat использует
  tempo-aware и beat-jitter negatives, Soft-DTW для onset/contact и OT/EMD для
  bar-level accent mass. Это мотивирует наши beat-jitter, within-bar и
  phrase-level controls, но сама paper-метрика не является автоматически
  валидированным critic для native LightSem body. [MotionBeat](https://arxiv.org/abs/2510.13244).
* **Музыкальные признаки.** MuQ — допустимый frozen audio representation,
  однако его наличие не доказывает, что downstream body использовал музыку;
  нужны counterfactual audio interventions. Pinned input contract проекта
  остаётся MuQ layer 9 + Base92, 24 kHz/25 Hz MuQ и явные timestamps.
  [Официальный MuQ repository](https://github.com/tencent-ailab/MuQ).
* **Оценка должна быть многомерной.** AIST++ и последующие работы обычно
  разделяют motion quality, diversity и music–motion correlation; эта схема
  полезнее одного composite score. [AIST++/AI Choreographer](https://arxiv.org/abs/2101.08779).

## Что уже измерено в проекте и нельзя переоткрывать как новую гипотезу

Vault recall поднимает отрицательные результаты раньше новых идей:

* На AIST++ beat-scale audio→kinetic-energy mapping имел почти нулевой
  inter-dancer ceiling (`r≈0.024` на 0.1 s), а held-out track Ridge был
  отрицательным даже после beat-synchronous segmentation. Структурные
  boundaries действительно содержали общий сигнал, но это слабый ceiling,
  который лучше получать из audio structure напрямую. Поэтому actor dance не
  используется как dense target для foundation light/body head.
* В RPMC audio→light persistence/history объяснял примерно `R²≈0.71`, а
  добавление audio поверх history дало отрицательный marginal gain (`≈-0.005`)
  в том frozen recipe. Значит любой новый predictive metric обязан показывать
  gain над motion/history baseline и time-shift null; абсолютный R² или
  audio-only fit вводит в заблуждение.
* Ранее были гипотезы, где высокие BAS/visible motion скрывали failed audio
  conditioning. Они остаются diagnostic-only; новый foundation audit должен
  измерять own-song diagonal advantage и сохранять same-BPM donor, beat-jitter
  и reused-body controls.

Это не запрещает новую foundation-гипотезу: она меняет объект с dense pose
supervision на conditional latent/body/light representation и тестирует
incremental information, а не повторяет старую training recipe.

2025–2026 работы с новым encoder/critic следует считать proposal, пока не
проверены код, checkpoint, split leakage, negative controls и human calibration.
Особенно нельзя без проверки переносить их BAS/FID numbers на 30 Hz native
22-joint body или на emitted DMX.

## MMCA: измерительная модель

MMCA выдаёт вектор evidence и gate-вердикт. Weighted average не используется
как основной verdict: высокий beat score не компенсирует отсутствие
music-conditioning или raw-body safety.

### 1. Raw safety и collapse (обязательный pre-gate)

Запускается существующий `scripts/evaluate_training.py` на сохранённом raw
30 Hz body, до mapper/interpolation/DMX/video. Сохраняются finite fraction,
root-relative speed/acceleration/jerk, root/support, inactive frames,
representation/sample rank, duplicate streams и provenance. Любой raw safety
failure quarantine-ит результат; downstream smoothing/D4/D6 не пересчитывает
его в pass.

### 2. Beat synchrony

Из body строится root-relative joint-speed energy `e(t)` и event list из raw
local maxima с явным prominence/refractory threshold. Для beats `b_i` считаются:

* BAS: `mean_i exp(-d(event_i, beat)^2 / (2 sigma^2))`;
* metronome upper reference: event на каждом beat;
* circular-shift null: та же event train после случайного circular shift;
* circular phase concentration `R`/Rayleigh `z` и signed phase offset;
* lag scan между audio onset envelope и `e(t)`.

Отчёт всегда хранит BAS, metronome, null mean/sd/z и event count. BAS без этих
нулей не считается evidence.

### 3. Accent–kinetic coupling

Музыкальный onset/accent и motion energy агрегируются в одинаковые beat bins.
Считаются rank correlation и circular beat-bin null. Это отделяет «движусь с
периодом» от «ставлю больше энергии туда, где меняется акцент». Нормировка
внутри каждого track допустима только для сравнения формы; абсолютная энергия
остаётся отдельным physical diagnostic.

### 4. Такт и фраза

Внутри каждого 4-beat bar строятся нормированные vectors `[accent_1..4]` и
`[energy_1..4]`; считаются cosine и rank correlation. Далее нужен phrase layer:
MuQ/Base92 self-similarity/novelty boundaries и beat-synchronous motion
self-similarity сравниваются через constrained DTW/recurrence agreement. Это
нельзя заменить global BPM correlation: repeated one-bar loop должен получить
низкий phrase novelty/recurrence score, даже если BAS высок.

### 5. Музыкальная причинность (anti-reuse gate)

Для минимум двух собственных song inputs при одинаковых weights, seed/noise,
non-audio inputs, mapper и clock строится полная матрица `M[i,j]`:

`M[i,j] = synchrony(body generated from song i, audio of song j)`, где
engineering evidence matrix усредняет BAS, beat-bin accent correlation и
within-bar cosine после приведения их к сопоставимой шкале; отдельная BAS
матрица сохраняется, чтобы видеть механизм.

Требуется diagonal advantage:

`mean(diag(M)) - mean(offdiag(M)) >= tau_conditioning`.

Одновременно нужны:

* same-seed audio swap / beat-jitter / time-shift response в root-relative body,
  а не только в RGB;
* donor-audio и shuffled-audio controls;
* похожая BPM пара, чтобы BPM shortcut не прошёл;
* frozen-body control: смена soundtrack не должна создавать новый body задним
  числом;
* limb intervention: изменение входной limb/teacher должно менять только
  assigned body/light channels согласно attribution map.

Не путать response с musicality: response необходим, но arbitrary noise-like
change всё равно провалит diagonal/phrase gates.

### 5a. Foundation latent diagnostics

До оценки body/light heads сохраняется latent tensor и проверяется отдельно:

* spectral effective rank, constant-channel fraction, temporal delta RMS и
  pairwise cosine; это различает latent collapse и output-head collapse;
* frozen linear probes from MuQ-only, Base92-only and MuQ+Base92 to latent
  residual/future motion, all evaluated with grouped song folds;
* conditional permutation importance: permute MuQ time blocks, beat phase,
  Base92 scalar groups and fixture metadata one family at a time, holding seed
  and all other inputs fixed;
* if a learned loss projector is present, audit both native latent and projected
  latent. A healthy projected space cannot rescue a collapsed native trunk.

The most useful foundation statistic is incremental, not absolute: blocked
future-motion predictive gain from audio + motion history versus motion history
alone, with a circular audio-shift null. This is the CPU proxy implemented in
`audio_predictive_gain`; call it a conditional-information proxy, never an
unbiased CMI estimate.

### 6. Actor reference

Актёрский body — calibration distribution, а не framewise target: dance is
one-to-many. На actor/generated считается один и тот же descriptor vector:
event density, phase offset/R, beat-bin accent profile, within-bar profile,
root-relative limb-energy proportions, hold/transition distribution,
velocity/acceleration/jerk and recurrence descriptors. Generated result должен
попадать в held-out actor percentile bands и быть дальше от shuffled-audio/null
контролей, чем от real split-half floor. Для style можно добавить MDSC-like
frozen cross-modal encoder только после split-by-song calibration; его score не
может заменять raw kinematic checks.

### 7. Body-to-light retention

Те же timing/structure descriptors считаются на decoded moving-head angular
streams and emitted intensity events. Сравниваются body→requested→decoded DMX
lag, event recall, recurrence agreement и attribution retention. Color/RGB и
brightness отчёты остаются отдельными. Идентичные output streams — один
representative плюс явный duplicate diagnostic, не «несколько движений».

## Conservative machine verdict

Рекомендуемые статусы:

* `failed_safety`: nonfinite/raw dynamic limit/invalid provenance;
* `collapsed_motion` или `output_collapsed`: inactivity/duplicates/rank,
  отдельно от zero event count;
* `unconditioned_or_shared_prior`: diagonal margin отсутствует или аудио
  intervention не даёт root-relative response;
* `beat_locked_only`: BAS выше shift-null, но accent/bar/phrase layer не
  проходит (типичный metronome cheat);
* `unqualified_musicality`: нужные song/actor/control evidence отсутствуют;
* `diagnostic_music_aligned`: все engineering gates прошли. Это ещё не
  художественная acceptance; нужен human calibration set и blind review.

Thresholds должны быть объявлены до scoring и храниться в manifest. Начальные
значения (`BAS null z>=2`, diagonal margin `>=0.03`, within-bar cosine
`>=0.60`) — engineering priors для smoke, не literature constants. Их нельзя
подбирать по candidate output; после калибровки на actor/AIST++ требуется новый
protocol id.

## Falsifiable claims for the next study

1. **Audio dependence.** If native generator uses music, replacing audio with a
   same-BPM donor or beat-jitter control while holding seed/noise fixed reduces
   diagonal own-song margin and changes root-relative future motion.
2. **Beat-only cheat.** A metronome or one-bar oscillator raises BAS but does not
   raise beat-bin accent null-z, within-bar profile, phrase novelty or actor
   percentile coverage.
3. **Foundation repair.** A repaired checkpoint may be promoted only if raw
   safety/collapse passes on both songs, own-song diagonal wins, and its
   body-to-DMX retention is above the frozen baseline. A lower training loss or
   nonzero video motion is insufficient.
4. **Actor calibration.** Real actor split-half distance defines the attainable
   descriptor floor; a generated clip closer to this floor than shuffled/donor
   controls is evidence of distributional plausibility, not proof of identical
   choreography.

## First implementation

`lightsem.eval.music_motion` and `scripts/evaluation/audit_music_motion.py`
implement the CPU core: raw root-relative motion profile/events, BAS with
metronome and shift null, continuous onset/energy lag scan, beat-bin coupling,
within-bar alignment, beat-synchronous phrase recurrence/novelty, the full
cross-song conditioning matrix, latent effective rank diagnostics, actor
distribution calibration and a blocked ridge `audio_predictive_gain` proxy.
Decoded-DMX retention remains the next downstream stage.

The synthetic benchmark
`docs/research/2026-10-19-music-motion-framework/METRIC_BENCHMARK_V5.json` tested
accent-conditioned motion, a BPM-only metronome, a reused body, root-only drift
and collapsed/healthy latent tensors. It showed:

* the composite own-song matrix separated accent-conditioned motion (`margin
  ≈0.266`) from both metronome and reused-body controls (`margin=0`);
* latent effective rank separated healthy and constant latent tensors, but did
  not establish audio use;
* root-relative motion correctly made root-only drift inactive;
* BAS was identical for the accent-conditioned and BPM-only controls, so BAS
  alone is rejected;
* the scalar predictive-gain probe was unstable on periodic synthetic inputs;
  it remains secondary. The train-only-PCA `feature_family_predictive_gain`
  probe is the path for comparing Base92 against MuQ layers, but it still needs
  real song-split calibration before becoming a gate. In this synthetic run,
  the simple and advanced feature-family probes both responded, but the
  advanced layer-like features did not beat the simple features; this is a
  warning against assuming a deeper MuQ layer is better without a frozen
  song-split comparison.
* phrase recurrence/novelty separated the accent-conditioned synthetic body
  (`recurrence null-z≈2.4–2.9`) from the weaker metronome/reused-body controls;
  it is useful as an advanced secondary gate, not a replacement for causal
  audio interventions.
* actor calibration correctly reports distributional distance, but it needs
  multiple real actor references and a split-by-song floor; it is not a
  choreography target or a proof of music conditioning.
