# Что в старом LightSem действительно меняет танец и свет

Дата отчёта: 27 октября 2026 года  
Объект: старый fixed-C pipeline LightSem, native dance-сеть и body-to-light / LightLayers сеть  
Режим: frozen checkpoint, один seed, причинные вмешательства, без продвижения checkpoint

## Итог

Входы действительно доходят до результата, но разные части системы отвечают за
разные выходы.

1. **Native dance-сеть реагирует прежде всего на Base92 и MuQ.** На свежем
   waveform-пересчёте Shpongle выключение всей Base92-группы изменило
   root-relative тело на **0.025750 м**, выключение MuQ — на **0.011360 м**.
   Base92 даёт больший эффект в этом рецепте.
2. **Надёжное двухпесенное ядро Base92** — `onset_ema`, `beat_period_s` и
   `mel_63`. Они единственные из 92 каналов, которые превысили измеренный
   двухсигмальный порог одновременно на Shpongle и AURORA — Cure For Me.
3. **Это ещё не доказательство, что каждый такой канал имеет уникальный
   музыкальный смысл.** Нулевое маскирование уводит вход за пределы train
   distribution, а Base92-каналы коррелируют друг с другом. Результат означает
   «выход чувствителен к этому каналу в данном вмешательстве», а не «канал
   является единственной причиной хореографии».
4. **Явные source-группы и timing-группы почти не двигают тело.** Отдельные
   drums/bass/other/vocals и mixture/beat/downbeat/section остаются внутри или
   рядом с noise floor.
5. **Body-to-DMX mapper сохраняет информацию о конечностях.** Патч левой руки
   двигает назначенные fixtures сильнее неназначенных.
6. **LightLayers использует аудио для палитры, RGB и emitted luma.** При
   фиксированном теле эти изменения не доходят до pan/tilt и dimmer bytes.
7. **Музыкальная хореография в сильном смысле пока не доказана.** Есть
   чувствительность к audio features и cached audio identity, но нет
   полноценного двухпесенного waveform-backed теста с phrase/motif/hold
   timing для обеих песен.

## Как проводился аудит

```text
song -> MuQ / Base92 / descriptors / stereo context
     -> native C -> raw body 30 Hz
     -> mapper / LightLayers -> decoded DMX 40 Hz -> RGB, luma, pan/tilt
```

В каждой causal-матрице фиксировались checkpoint, seed/noise, clock, non-audio
inputs, stereo context, mapper и renderer. Сначала измерялся raw 30 Hz body;
mapper, interpolation, DMX и video не использовались до прохождения raw-safety
и collapse gates.

Основная диагностическая база:

- root-relative RMS noise floor: **σ = 0.003514 м**;
- standardized-effect noise floor: **σ ≈ 0.01157**;
- raw safety: finite values, speed/acceleration/jerk ratio к full-arm ниже
  **2×**;
- collapse: native effective rank **139**, отсутствие constant body,
  inactive frames и exact duplicate streams;
- deterministic repeat: два одинаковых forward-pass с byte-identical native
  output.

Нулевой event count, разные hash или ненулевая скорость сами по себе не
доказывают ни collapse, ни музыкальную обусловленность.

## 1. Native dance-сеть

### Семейства MuQ и Base92

Cached two-song screen на 60-секундных окнах дал:

| Вмешательство | Shpongle | AURORA — Cure For Me | Вывод |
|---|---:|---:|---|
| Zero MuQ | 0.01139 м; standardized 0.03788 | 0.03371 м; 0.11064 | Механически влияет на тело; Shpongle ниже старого MDE 0.03 м |
| Zero Base92 | 0.02580 м; 0.08584 | 0.05845 м; 0.19184 | Более сильная семейная чувствительность в обоих cached окнах |
| Zero source descriptors | 0.000887 м | 0.001283 м | Слабый эффект, около noise floor |

Все эти cached arms были конечными, rank 139, повторяемыми и raw-safe. Но
cached-only run не являлся полноценным own-song waveform test: исходные
hash-bound waveforms отсутствовали.

### Проверка `/root/shpongle.wav`

Пользовательский WAV был обработан как отдельная versioned source binding:

- source SHA: `d547ea832141221ed24b0cc6b24bb94e24316e6fe49f443efd27c96b9b5de603`;
- длительность: **277.307 с**;
- analyzed waveform SHA: `6cc158b1eece2527c4d7032494045c0f222222634c2ee77cc45d59042fdd8e6c`;
- свежий timeline SHA: `ddb392b717abf376f1d32ae1c0fde50a58a96d5c0ccf51d9bc62e7e41b266e93`;
- свежий Base92 SHA: `adc3ab2289c2cfb525f5df3a6a34f2486a0c8e5dd67294c03b0eb8d5431d3482`.

Результат почти совпал с cached Shpongle:

| Arm | Свежий WAV | Cached | Разница |
|---|---:|---:|---:|
| Zero MuQ | 0.011360 м | 0.011387 м | 0.000027 м |
| Zero Base92 | 0.025750 м | 0.025804 м | 0.000054 м |

Full bodies имеют world-position RMS difference **0.000324 м** и correlation
**0.99999986**. Все четыре свежих arm-а конечны, active, rank 139,
repeat-stable и raw-safe; худшие ratios speed/acceleration/jerk равны
**1.040× / 1.103× / 1.119×**.

Это подтверждает пригодность WAV-конверсии для Shpongle causal screen. Она не
является восстановлением старого hash-bound файла и не заменяет проверку Cure
For Me.

### Все 92 Base92-фичи на свежем Shpongle waveform

Каждый из 92 каналов был выключен отдельно. Топ однопесенных эффектов:

| Канал | Root-relative RMS |
|---|---:|
| `mel_50` | 0.010273 м |
| `mel_47` | 0.009184 м |
| `onset_ema` | 0.008552 м |
| `mel_63` | 0.008483 м |
| `mel_58` | 0.007987 м |
| `beat_period_s` | 0.007534 м |

Рейтинг всех 92 свежих эффектов совпал с cached Shpongle-картой: Pearson
**0.9999997**, Spearman **0.9999846**, MAE **3.43×10⁻⁶ м**.

Все 92 arm-а прошли finite/active/rank/repeat gates. Максимальные raw ratios:
speed **1.117×**, acceleration **1.388×**, jerk **1.991×**. Канал `mel_50`
почти достигает 2× quarantine boundary; его нельзя использовать для красивого
демо без отдельного safety review.

Шесть каналов выше двух сигм на Shpongle — это one-song screen. Для
двухпесенного утверждения нужно пересечение с Cure For Me; оно остаётся:

```text
onset_ema, beat_period_s, mel_63
```

`mel_47`, `mel_50` и `mel_58` сильны на Shpongle, но пока не имеют двухпесенной
поддержки.

### Почему нельзя назвать все 92 каналами «причинами»

Zero-mask меняет не только выбранную координату, но и распределение входа:
модель видит невозможный для обучения нулевой канал. Поэтому большой эффект
может означать OOD/distribution-shift response. Это проверено support-matched
заменой на checkpoint TRAIN mean:

- strongest Shpongle: `rolloff` — **0.007175 м**;
- strongest Cure For Me: `mel_63` — **0.004410 м**;
- ни один канал не превысил двухсигмальный порог одновременно в обеих песнях.

Mean-replace не доказывает, что Base92 не используется; он запрещает называть
каждый zero-mask effect уникальной семантикой.

### Source и timing группы

Раздельное masking drums/bass/other/vocals дало **0.000003–0.000697 м** на
Shpongle и **0.000400–0.000825 м** на Cure For Me; максимальный standardized
effect — **0.00232**.

Timing-группы (mixture, beat, downbeat, section) дали максимум **0.00293 м** на
Shpongle и **0.00648 м** на Cure For Me; conservative standardized maximum —
**0.00973**.

Обе группы механически присутствуют в computation graph, но на этой native-body
шкале слабы и не проходят screen MDE. Это не означает, что timing не может
влиять на phrase-level metric; такой тест ещё не выполнен на полном waveform.

### Cached donor и time-shift

Замена audio stream целиком на cached stream другой песни дала **0.03767 м** и
standardized effects **0.1253 / 0.1236**. Но Shpongle target ← Cure donor
небезопасен: speed **1.575×**, jerk **2.033×**. Этот arm quarantined и не
рендерился.

Alpha=.75 interpolation стала raw-safe в обеих направлениях и дала
standardized effects **0.1044 / 0.0961**, выше 15-секундных circular-shift
null-ов **0.0372 / 0.0859**. Это сильное cached audio-identity evidence, но не
доказательство phrase choreography: waveform-backed own-song donor controls
ещё не закрыты.

## 2. Body-to-DMX mapper

Патч root-relative левой руки перед mapper-ом дал:

- assigned fixtures: **3.91° RMS**;
- unassigned fixtures: **1.88° RMS**;
- ratio: **2.08×**;
- duplicate fixture rate: **0.0**.

Это механическое доказательство сохранения source-limb information:
назначенные приборы действительно получают больше сигнала от соответствующей
конечности. Это не доказательство того, что аудио само породило эту конечность,
и не доказательство художественной хореографии.

## 3. LightLayers и lighting output

### Palette-state seam

На cached 94-D audio bus изолированное masking изменило palette states:

| Группа | Shpongle | Cure For Me |
|---|---:|---:|
| mel | 0.567 | 0.334 |
| energy | 0.402 | 0.992 |
| timbre | 0.013 | 0.663 |
| rhythm | 0.834 | 0.987 |

Это подтверждает audio dependence LightLayers palette logic. Само по себе оно
не доказывает изменение тела, DMX или brightness.

### Fixed-body endpoint через P2

При фиксированных body и mapper audio groups дошли до RGB и emitted luma:

- decoded RGB maximum: **0.2694** 8-bit RMS Shpongle, **0.4110** Cure For Me;
- emitted-luma RMS: Shpongle **0.00628–0.01603**, Cure For Me
  **0.00016–0.00315**;
- decoded angular pan/tilt: **0.0°**;
- dimmer-byte RMS: **0.0**;
- DMX finite, duplicate fixture rate: **0.0**.

Emitted luma = dimmer × max(RGB). Поэтому неизменный dimmer byte не означает
неизменную воспринимаемую яркость. В текущей fixed-body ветке аудио меняет
цвет и luma через RGB, но не двигает головы и не меняет raw dimmer.

### Opposite-song LightLayers donor

Cached donor audio при фиксированных body/mapper дал:

- Shpongle target + Cure donor: RGB **0.47458**, luma **0.007761**;
- Cure target + Shpongle donor: RGB **0.42122**, luma **0.014428**;
- pan/tilt и dimmer: ровно **0** в обеих направлениях.

Это causal output-stage evidence для цвета и emitted luma. Это не evidence для
audio-to-pan/tilt, body choreography или full-waveform conditioning.

## Что реально работает

| Компонент | Статус | Что можно утверждать |
|---|---|---|
| Native C + Base92 | Подтверждено как чувствительность | Base92 меняет raw body; устойчивое двухпесенное ядро: `onset_ema`, `beat_period_s`, `mel_63` |
| Native C + MuQ | Подтверждено как механическое влияние | MuQ меняет body, но эффект recipe-conditional и меньше Base92 на Shpongle |
| Audio identity donor | Направленно подтверждено | Cached audio stream меняет body, но один donor unsafe; alpha=.75 raw-safe |
| Body→DMX limb assignment | Подтверждено механически | Assigned fixtures сильнее реагируют на assigned limb |
| LightLayers palette | Подтверждено | Audio groups меняют palette state в обеих песнях |
| LightLayers RGB/luma | Подтверждено при fixed body | Audio groups доходят до RGB и emitted luma |

## Что не работает или пока не доказано

| Утверждение | Статус | Причина |
|---|---|---|
| Все 92 Base92 имеют независимый музыкальный смысл | Не доказано | Корреляция каналов и OOD zero-mask; mean-replace даёт powered null |
| Explicit drums/bass/other/vocals сильно управляют телом | Не подтверждено | Эффекты внутри noise floor |
| Beat/downbeat/section masks уже доказывают phrase timing | Не доказано | Native RMS слишком мал; phrase/motif/hold analysis отсутствует |
| Audio напрямую двигает pan/tilt в fixed-body LightLayers path | Не подтверждено | Угловой эффект ровно 0° |
| Audio напрямую меняет dimmer bytes | Не подтверждено | Byte-level effect ровно 0; luma меняется через RGB |
| Cached donor доказывает музыку в полном генераторе | Не доказано | Waveform отсутствовал, один donor unsafe, time-shift controls обязательны |
| Шпонгле WAV — это старый hash-bound original | Нет | SHA отличается; это отдельная versioned binding |
| Cure For Me full-waveform validation закрыта | Нет | Авторизованный exact waveform не найден |
| Checkpoint или demo можно продвигать | Нет | Causal screens diagnostic-only; choreography и phrase gates не закрыты |

## Происхождение пропавших waveform

Старые fixed-C bindings ожидают Shpongle SHA `95e438…` и Cure For Me SHA
`5363ce…`. Их точные файлы не найдены ни в workspace, ни в `/mnt`-кэшах;
relocation map содержит только более старые несовпадающие hashes. `/root/shpongle.wav`
имеет SHA `d547ea…`, поэтому он доказал пригодность WAV-конверсии, но не может
быть молча переименован в прежний оригинал. Для Cure For Me найденный
`/root/4 - Cure For Me.m4a` также имеет другой SHA и не был подставлен.

## Что дали найденные papers

Литература не заменила эксперименты, но изменила правильный protocol.

- **FANS** (arXiv:2402.08845) — necessity/sufficiency в conditional
  neighborhood; следующий шаг для локальных Base92 и lighting-bus replacements.
- **MOSPA** (arXiv:2507.11949v2) — разделять timbre, event, tempo и style;
  нельзя смешивать palette, motion и brightness в одном score.
- **RAVEL / MDAS / Same Task–Different Circuits** — проверять target Cause,
  off-target Isolate и functional interchange, а не только величину patch delta.
- **CausalMM, MCD, DIME/CMI** — Jacobian, CMI, CKA и attention routing годятся
  для screening, но требуют matched forward intervention для причинного вывода.
- **EDGE, MoMu-Diffusion, Listen-Denoise-Action, DanceMeld, Lodge,
  TempoMoE, MotionBeat** — нужны отдельные проверки phrase, primitive, accent,
  contact и hold; простой beat correlation недостаточен.
- **DiscoForcing** (arXiv:2605.28491) — полезное разделение rhythmic code и
  periodic phase; его можно применить к Base92 rhythm/beat controls с causal
  latency и raw-body safety.
- **SeqLight** (arXiv:2605.03660) — отделяет global hue/value от
  venue-specific fixture decomposition; подтверждает раздельные RGB/luma и
  angular-DMX endpoints.
- **Glow with the Flow** (arXiv:2602.08838) — разделяет event, section и
  palette effects; это rule-based источник, не доказательство moving-head
  causality.
- **Motive** (arXiv:2601.08828) — motion-weighted training-data attribution;
  полезен для будущего retraining audit, но не является frozen-input causality.

Общий вывод papers: attribution нельзя получать из красивого видео,
correlation, Jacobian или reconstruction quality. Нужны matched retain/remove,
conditional controls, raw safety и отдельные output endpoints.

## Следующие необходимые эксперименты

1. Получить или отдельно авторизовать Cure For Me waveform и прогнать свежие
   MuQ/Base92 features, затем все 92 Base92-канала тем же протоколом.
2. Для Shpongle и Cure повторить Base92 с support-matched local/mean
   replacements; zero-map оставить только как sensitivity screen.
3. Добавить phrase/motif/hold/tempo-aligned metrics и circular-shift, same-BPM
   donor и beat-jitter controls.
4. Прогнать LightLayers retain/remove через настоящий fixed mapper и emitted
   DMX с body fixed; сохранять отдельно pan/tilt, RGB, dimmer, luma, events,
   saturation и duplicate streams.
5. Для regenerated-body tests проверять source-body provenance и raw 30 Hz
   safety до любого DMX/video. Небезопасный arm не сглаживать и не скрывать
   fixture caps.
6. Только после этих gates решать, есть ли основание для полноценного demo,
   checkpoint promotion или retraining.

## Артефакты

- [Полный research memo на английском](/root/lightsem/docs/research/2026-10-26-causal-feature-influence/RESEARCH_MEMO.md:1)
- [Causal feature map](/root/lightsem/docs/research/2026-10-26-causal-feature-influence/CAUSAL_FEATURE_MAP.md:1)
- [Converted family receipt](/mnt/lightsem-data/experiments/lightsem/shpongle_converted_audio_recompute_20261027_v1/causal_family_matrix60/CAUSAL_RECEIPT.json)
- [All-92 converted receipt](/mnt/lightsem-data/experiments/lightsem/shpongle_converted_audio_recompute_20261027_v1/BASE92_CONVERTED_RECEIPT.json)
- [All-92 converted CSV](/mnt/lightsem-data/experiments/lightsem/shpongle_converted_audio_recompute_20261027_v1/BASE92_CONVERTED_SUMMARY.csv)
- [Cached all-92 two-song receipt](/mnt/lightsem-data/experiments/lightsem/dance_base92_per_feature_cached_only_20261101_v1/PORTFOLIO_RECEIPT.json)
- [Cached mean-replacement comparison](/mnt/lightsem-data/experiments/lightsem/dance_base92_per_feature_mean_cached_only_20261101_v1/BASE92_ZERO_MEAN_COMPARISON.csv)
- [Cached LightLayers RGB/luma endpoint](/mnt/lightsem-data/experiments/lightsem/light_audio_group_dmx_causal_cached_20261028_v5/PORTFOLIO_RECEIPT.json)
- [Cached LightLayers donor endpoint](/mnt/lightsem-data/experiments/lightsem/light_audio_donor_causal_cached_20261030_v2/PORTFOLIO_RECEIPT.json)

Все результаты записаны в Vibescience problem
`lightsem-causal-feature-influence-20261026`. Ни один checkpoint, body или
lighting output не promoted по одному этому отчёту.
