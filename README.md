# LightSem

LightSem turns audio into stage lighting. The **motion baseline is repaired native C (HF16, seed0)**, selected on October6, with frozen MuQ in every run. The **color baseline remains P2**: independent Rich96 (`I_VR_P0`), palette history, 300 ms transitions and two spatial groups. D4-max motif composition and D6 external pacing are independently optional. Motion comes from the own-song body through the fixed01 mapper; autonomous sinusoidal choreography is rejected. Simulation fixture speed/acceleration limits remain off. Both Shpongle and Cure For Me supply their own trajectories in the verified comparisons.

## Selected baseline CLI

The active definition is [configs/dance/fixed_c.json](configs/dance/fixed_c.json).
It pins the qualified C checkpoint and mandatory MuQ encoder. The older
[dance reference config](configs/dance/baseline.json) remains unchanged because
completed experiment receipts bind its original bytes. Model weights, datasets
and movies remain external assets; restore the documented data/model mount or
use `--checkpoint`, `--d6-checkpoint`, `--features` and `--base92` to relocate
the exact authenticated assets. There is no alternate-encoder fallback.

```bash
# Default: fixed C + MuQ, P2 color and restored historical brightness.
bash run_torch.sh -m lightsem.cli render --wav /root/shpongle.flac --out runs/fixed_c.mp4

# Independent optional modifiers; D6 also works without D4.
bash run_torch.sh -m lightsem.cli render --wav /root/shpongle.flac --out runs/c_d4.mp4 --d4max
bash run_torch.sh -m lightsem.cli render --wav /root/shpongle.flac --out runs/c_d6.mp4 --d6
bash run_torch.sh -m lightsem.cli render --wav /root/shpongle.flac --out runs/c_d4_d6.mp4 --d4max --d6

# Export the complete body and provenance without rendering.
bash run_torch.sh -m lightsem.cli dance --wav /root/shpongle.flac --out runs/c_body
```

`render --start 90 --duration 60` crops only the rendered interval; analysis
and body generation still use the complete song. Audio speed stays unchanged.
`--beams`, `--lasers` and `--bulbs` change the output counts (defaults8/5/50).
The videos show the 3D body, actual emitted DMX and unfiltered movement traces.
New audio uses the pinned MuQ/HTDemucs/BeatThis/All-In-One extraction stack;
`--audio-python` and `--audio-device` select its installed environment/device.
`--audio-cache` selects the whole-song feature cache mount; its default is
`/mnt/lightsem-data/cache/native_c_cli`. Extraction preserves the 20 GiB free
space floor and does not delete data automatically.
New-song P2 inference starts from fixed black RGB before its first generated
cue; the two registered showcase tracks retain their exact accepted streams.

D4-max and D6 modify a generated trajectory; they do not retrain C. The
selected weights were trained on the bounded72-song Just Dance cohort, and
do not yet include the subsequently corrected quiet-loss gradient. Broader
data training remains an unexecuted next recipe. Raw stability passed; musical
pacing superiority is not established. Completed experiments and negative
verdicts are preserved in the [campaign closure](docs/research/2026-10-05-native-dynamics/CAMPAIGN_CLOSURE.json).
The former rule-driven `render` is available only as `legacy-render` for
reproducing historical studies.

The [September22 campaign](experiments/2026-09-22-color-beat/README.md) contains the training implementation and earlier measurements. The user selection is a product-development decision from Shpongle review; previous scientific results retain their recorded scope. The older causal `LightModel` API in `lightsem/model/infer.py` is a separate implementation, not a drop-in loader for the independent light-layer model.

The [visual-learning studies](experiments/2026-09-29-visual-next/README.md) are complete under the automated exploratory scope: [19 Shpongle review videos](experiments/2026-09-29-visual-next/records/REVIEW.md) and [measured results](experiments/2026-09-29-visual-next/records/RESULTS_EXPLORATORY.md). Source annotation is optional; original independent-label hypotheses remain unconfirmed. Baseline C remains unchanged.

## Как устроен код `lightsem/`

Ниже — карта исходников для чтения проекта. Пакет называется **LightSem**, его
каталог — `lightsem/`. Названия двух подкаталогов в коде: `model/` (единственное
число) и `eval/` (evaluation, оценка), а DMX-каталог записан как `dmx/`.
Это несколько поколений системы, а не одна модель, которую последовательно
вызывают все папки.

### Общая схема

| Путь | Что происходит |
| --- | --- |
| Ранний свет по правилам | `audio/features + structure + anticipation → dmx/scenes → dmx/universe → sim/renderer`; этот исторический путь вызывает `cli.py legacy-render` |
| Свет из описаний движения | `audio/online → model/light_trunk + infer → conductor/v0 или hybrid → dmx/encode → sim`; модель предсказывает характеристики движения, а не позы тела |
| Независимые световые слои | Аудиопризнаки → `model/light_layers/` → предсказания акцентов, цветовых событий и палитры → `overlay.py` и переходы → RGB в DMX; здесь реализовано семейство моделей выбранного цветового P2 |
| Перенос танца на N выходов | Сохранённое тело `(T, 22, 3)` → `conductor/body_transfer` или `body_transfer_v2` → компиляция в DMX → декодирование → `sim/body_stage_renderer` или `lit_body_renderer`; P2 поставляет цвет отдельно |
| Большая фестивальная сцена | Исходный DMX → `festival/patch` → несколько DMX-universe → Blender; также есть передача и приём Art-Net |

`T` — число кадров. DMX-universe — блок из 512 управляющих каналов; канал не
равен прибору: один прибор занимает несколько каналов. Art-Net переносит эти
кадры по UDP. Конкретный запуск, checkpoint и состав слоёв задаются скриптом
кампании и конфигурацией; расположение файла само по себе не означает, что его
используют в текущем демо.

### `audio/` — перевод звука в числовые признаки

На входе аудио; на выходе временные ряды энергии, спектра, атак звука, ритма
и структуры. Здесь анализируют музыку, но не управляют приборами.

- [`features.py`](lightsem/audio/features.py): офлайн-анализ трека — beats,
  предполагаемые начала тактов, onsets, энергетические и спектральные признаки
  в `AudioFeatures`. Начала тактов определяются эвристикой.
- [`online.py`](lightsem/audio/online.py): `AudioBus` с log-mel спектром, RMS,
  flux отдельно в низких и высоких частотах и другими каналами на сетке
  управления. Каузальные расчёты используют прошлое звука; beat-каналы могут
  приходить из офлайн-трекера и помечаются `oracle_beat`. Название `online`
  само по себе не гарантирует, что весь вход пригоден для живого режима.
- [`beatgrid.py`](lightsem/audio/beatgrid.py): подбор постоянного темпа и фазы
  по onset-сигналу для корпусных целей; это не универсальный live-трекер.
- [`structure.py`](lightsem/audio/structure.py): поиск границ частей песни
  и эвристические метки `intro`, `build-up`, `drop`, `breakdown`, `outro`.
- [`anticipation.py`](lightsem/audio/anticipation.py): напряжение и время до
  предполагаемого drop. Режим `causal` прогнозирует из прошлого; `oracle`
  читает известные будущие границы и служит эталоном для оценки.
- [`rhythm_clock.py`](lightsem/audio/rhythm_clock.py): отдельный офлайн-анализ
  ритма с локальной коррекцией периода и уверенностью.
- [`representation_rhythm.py`](lightsem/audio/representation_rhythm.py):
  диагностическая подмена пяти timing-каналов Base92 результатами пяти
  замороженных rhythm heads. Не обучает их заново и не выдумывает пропущенные
  биты; это экспериментальная политика, не автоматически принятый детектор.

### `conductor/` — перевод движения и событий в управление светом

Это слой соответствий: какая часть движения управляет яркостью, направлением
или шириной луча. Основные реализации детерминированные.

- [`v0.py`](lightsem/conductor/v0.py): `MotionDescriptors → параметры приборов
  → DmxStream`. Количество движения задаёт интенсивность, боковое положение —
  распределение света и pan, вертикальное движение — tilt/яркость, ширина позы —
  zoom, шаги и прыжки — акценты. Прямого аудиовхода у `ConductorV0` нет.
- [`hybrid.py`](lightsem/conductor/hybrid.py): исторический гибрид; сохраняет
  огибающие движения, но привязывает быстрые акценты к аудио beats/onsets.
- [`beat_trigger.py`](lightsem/conductor/beat_trigger.py): простой аудио-контроллер
  по правилам для сравнений, без танцора и нейросети.
- [`rig.py`](lightsem/conductor/rig.py): каноническая расстановка приборов
  и роли; [`scalers.py`](lightsem/conductor/scalers.py) и `scalers.json`:
  масштабирование дескрипторов по статистикам корпуса.
- [`body_transfer.py`](lightsem/conductor/body_transfer.py): прямой перенос
  22-суставного тела на произвольное число лучей, виртуальных лазеров и лампочек.
  Выходы привязаны к конечностям или пространственному полю тела. Здесь же
  пересчёт временной сетки, перевод целей в pan/tilt, упаковка нескольких
  universe и обратное декодирование. Аудио и самостоятельного генератора
  движения этот API не принимает.
- [`body_transfer_v2/`](lightsem/conductor/body_transfer_v2/): 15 исследовательских
  вариантов переноса. `anatomical.py` описывает кости, сгибы и части тела;
  `relational.py` — отношения между частями; `composition.py` — деформации
  общей композиции. `common.py` и `skeleton.py` задают геометрию и контракты,
  `installation.py` переводит поле тела в N выходов, `diagnostics.py` проверяет
  перенос. Это детерминированные адаптации идей статей, а не воспроизведение
  их обученных сетей.

Старые `beat_trigger`, `hybrid` и движущиеся сцены остаются в коде как отдельные
реализации. Их автономные эффекты не разрешены как замена неудачному переносу
танца. У `TransferConfig` ограничения скорости включены по умолчанию, но в
текущей симуляции их требуется явно отключать согласно [AGENTS.md](AGENTS.md).

### `data/` — доступ к исходным корпусам

Это Python-загрузчики и подготовка входов. Корневой каталог [`data/`](data/)
с файлами данных — другая папка; большие корпуса также находятся в
`/mnt/lightsem-data/`.

- [`aist.py`](lightsem/data/aist.py): читает AIST++ keypoints, разбирает
  идентичность записи, контролирует пропуски и строит manifest. По умолчанию
  использует `keypoints3d_optim`; длительность позы определяет длину аудиопары.
- [`ingest.py`](lightsem/data/ingest.py): обёртка над `yt-dlp` для загрузки медиа.
- [`oslo.py`](lightsem/data/oslo.py): записи свободного танца Oslo — позиции
  маркеров, маски пропусков, аудиопризнаки и аннотированные сегменты. Это не
  полный суставной скелет каждого человека.
- [`oslo_audio.py`](lightsem/data/oslo_audio.py): сопоставляет исходные треки
  с временной шкалой DJ-микса, восстанавливает аудиомикс и проверяет совпадение.
- [`rpmc_remote.py`](lightsem/data/rpmc_remote.py): доступ к удалённому RPMC-L2
  HDF5 частями через HTTP Range, без полной загрузки корпуса.
- [`sync.py`](lightsem/data/sync.py): сохранённая **опровергнутая** гипотеза
  коррекции AIST audio-motion offsets. По умолчанию поправки не загружаются;
  использовать этот модуль как принятую синхронизацию нельзя.

### `dmx/` — описание приборов и управляющие байты

- [`fixtures.py`](lightsem/dmx/fixtures.py): `FixtureState`, `Fixture`, `Rig`,
  адреса и профили каналов. `club_rig()` — исходные 14 приборов: четыре wash,
  шесть moving heads, два blinder, strobe и projector.
- [`encode.py`](lightsem/dmx/encode.py): векторная упаковка параметров
  яркости, HSV/RGB, pan/tilt и прочих команд в `uint8 (T, 512)`;
  также содержит ограничитель скорости изменения параметров `slew_limit`.
- [`decode.py`](lightsem/dmx/decode.py): читает параметры обратно из байтов,
  вычисляет модель излучаемого света и яркость для диагностики.
- [`scenes.py`](lightsem/dmx/scenes.py): ранний `SceneEngine` — таблица
  музыкальных разделов, палитры и процедурные эффекты с beat-акцентами.
  Это самостоятельный путь по правилам, а не код P2 или переноса тела.
- [`universe.py`](lightsem/dmx/universe.py): `DmxStream`, генерация потока
  раннего scene engine, сохранение NPY и cue lists JSON/CSV, простой Art-Net sender.

DMX — формат результата. Он сам по себе не доказывает, что движения создала
модель или что они зависят от музыки.

### `eval/` — измерение результата и материалы для сравнения

- [`light_features.py`](lightsem/eval/light_features.py): извлекает из
  декодированного DMX частоты событий, изменения цвета, скорости pan/tilt,
  временной спектр и пространственную вариативность. Признаки изменений
  отделены от уровней яркости, чтобы постоянный красивый свет не выглядел
  динамичным только из-за своего цвета.
- [`bas.py`](lightsem/eval/bas.py): совпадение световых событий с beats;
  сравнение с метрономом и случайными круговыми сдвигами, анализ задержки.
- [`fid.py`](lightsem/eval/fid.py): расстояние Фреше между распределениями
  световых признаков, diversity и split-half ориентир естественной разницы.
  Здесь FID относится к световым признакам, а не к изображениям Inception.
- [`ab.py`](lightsem/eval/ab.py): выравнивание яркости, случайный порядок
  слепых A/B-пар, форма оценки и отдельный ключ соответствий.

Это общий набор метрик. Специальные проверки цветовых моделей находятся в
`model/light_layers/evaluation.py`, а проверки тела и экспериментальных
кампаний — в `lightsem/experiments/`. Одно хорошее число из `eval/` не заменяет
проверку коллапса, зависимости от входа и сырой 3D-динамики.

### `festival/` — большая сцена, патч и сетевой транспорт

- [`patch.py`](lightsem/festival/patch.py): создаёт конфигурируемую расстановку,
  распределяет адреса по нескольким universe, проверяет патч и экспортирует
  timeline. Его `fanout_dmx` размножает команды исходного рига; увеличение
  числа приборов не добавляет независимой информации о танце. В модуле
  сохранилась историческая маркировка C, она не означает выбор C вместо P2.
- [`artnet.py`](lightsem/festival/artnet.py): кодирование/декодирование ArtDmx,
  состояние universe, отправка на явно заданные адреса и воспроизведение timeline.
- [`router.py`](lightsem/festival/router.py): live-маршрутизатор из исходного
  канонического DMX в фестивальный патч.
- [`blender_live.py`](lightsem/festival/blender_live.py): приём Art-Net
  и применение команд к уже открытой Blender-сцене.
- [`blender_scene.py`](lightsem/festival/blender_scene.py) и
  [`geometry.py`](lightsem/festival/geometry.py): сборка площадки, сцены,
  приборов, навесов, гирлянд, окружения и применение DMX.
- [`realistic_scene.py`](lightsem/festival/realistic_scene.py) и
  [`realistic_geometry.py`](lightsem/festival/realistic_geometry.py): отдельная
  тёмная площадка в реальном масштабе с Cycles и пассивной геометрией.

Лазеры здесь виртуальные, проекторы используют симуляционный профиль.
Описание запуска и сцены: [FESTIVAL_SCENE.md](docs/FESTIVAL_SCENE.md).

### `model/` — несколько семейств обучаемых моделей

Название каталога объединяет разные задачи и несовместимые checkpoints:

- [`encoder.py`](lightsem/model/encoder.py): `MacroEncoder`, causal
  Transformer по аудиоэмбеддингам; предсказывает вектор световой сцены сейчас
  и на будущих горизонтах. [`data.py`](lightsem/model/data.py) подготавливает
  RPMC-L2 песни, окна и разбиение для этой задачи.
- [`light_trunk.py`](lightsem/model/light_trunk.py): causal Conformer с тремя
  головами — A: непрерывные характеристики движения, B: вероятностные оценки
  событий, C: генерируемый style residual через rectified flow.
  [`light_data.py`](lightsem/model/light_data.py) и
  [`jd_data.py`](lightsem/model/jd_data.py) собирают тренировочные тензоры
  и разбиения, в том числе для Just Dance.
- [`infer.py`](lightsem/model/infer.py): загрузка старого `LightModel`,
  нормализация аудио, предсказание `MotionDescriptors`, извлечение событий
  и вызов conductor. Этот путь не генерирует видимый скелет и не загружает P2.
- [`group_decoder.py`](lightsem/model/group_decoder.py): экспериментальный
  `GroupDecoder` выдаёт параметры групп приборов — интенсивность, палитру,
  тип/частоту эффекта, участие; `GroupEngine` реализует эффекты и пакует DMX.
  В нём есть процедурная библиотека эффектов, отдельная от переноса тела.
- [`light_layers/`](lightsem/model/light_layers/): независимая офлайн-ветка
  световых слоёв. `model.py` определяет shared/independent сети;
  `visual.py` извлекает наблюдения из видео с масками;
  `targets.py` строит цветовые цели, включая Rich96;
  `training.py` собирает данные, обучает и предсказывает с историей палитры;
  `evaluation.py`, `metrics.py`, `palette_metrics.py` оценивают результаты;
  `overlay.py` применяет слои к DMX, `transitions.py` делает переходы RGB.
  P2 — выбранная конфигурация этой ветки, закреплённая в
  [`baseline.json`](configs/light_layers/baseline.json), а не имя Python-класса.

Генераторы полного тела и текущие исследования native dynamics находятся
за пределами `model/`: в [`lightsem/experiments/motion_study/`](lightsem/experiments/motion_study/),
[`native_dynamics/`](lightsem/experiments/native_dynamics/) и
[`native_repair/`](lightsem/experiments/native_repair/). Поэтому чтение только
`model/` не объясняет происхождение всех тел в последних демо.

### `motion/` — геометрия и измерения реального движения

- [`skeleton.py`](lightsem/motion/skeleton.py): подготовка COCO-17 скелета —
  нормализация по длине торса, полу и ориентации, офлайн-фильтрация,
  производные, зеркалирование и пересчёт частоты кадров.
- [`descriptors.py`](lightsem/motion/descriptors.py): `Skeleton →
  MotionDescriptors`: количество движения, скорости головы/рук/ног,
  вертикальные колебания, ширина позы, стилевые характеристики и события
  шагов/прыжков/акцентов. Дескрипторы распределены по блокам A/B/C.
- [`pose2d.py`](lightsem/motion/pose2d.py): извлечение COCO-17 keypoints из
  видео, отслеживание человека, confidence/маски и перевод в общий путь
  дескрипторов. Координаты встраиваются в плоскость; метрическая глубина
  из 2D-наблюдений не восстанавливается.
- [`io.py`](lightsem/motion/io.py): сохранение/загрузка кэшей дескрипторов
  и аудиошины в NPZ с JSON-метаданными, вырезание аудиоокна для записи танца.

Это измерительный слой, а не генератор хореографии. Его COCO-17 представление
отличается от 22-суставного native тела в `body_transfer`. Сглаженные
дескрипторы также нельзя использовать вместо проверки сырого generated body.

### `sim/` — изображение из фактически выданных команд

- [`renderer.py`](lightsem/sim/renderer.py): `StageRenderer` — headless
  ModernGL/EGL-рендер исходного клубного рига, кадры и видео со звуком через
  ffmpeg. Параметры приборов читаются из DMX-байтов.
- [`shaders.py`](lightsem/sim/shaders.py): GLSL raymarching — лучи в дымке,
  световые конусы, пол, стены и конструкции.
- [`body_stage_renderer.py`](lightsem/sim/body_stage_renderer.py): та же
  тёмная сцена для произвольного набора декодированных body-transfer выходов,
  с отдельными светящимися лампочками и тонкими виртуальными лазерными лучами.
- [`lit_body_renderer.py`](lightsem/sim/lit_body_renderer.py): добавляет
  непрозрачное исходное тело из костей-капсул и головы-сферы; оно освещается
  командами сцены и объявленной постоянной подсветкой для читаемости.
  Поддерживает перекрытие объектов для камеры, но не отбрасываемые тени
  и переотражения.

Рендерер показывает команды; он не должен придумывать движение, чтобы
замаскировать неподвижный или плохой выход модели. `sim/` и Blender-путь
`festival/` — два разных способа визуализации.

### В каком порядке читать

Для **текущего переноса тела**: `conductor/body_transfer.py` →
`body_transfer_v2/__init__.py` и `installation.py` → `sim/body_stage_renderer.py`
→ `sim/lit_body_renderer.py`; затем источник тела в нужной экспериментальной
кампании и отдельная цветовая ветка `model/light_layers/`.

Для **старого пути обучения на характеристиках танца**:
`motion/skeleton.py` → `motion/descriptors.py` → `audio/online.py` →
`model/light_data.py` → `model/light_trunk.py` → `model/infer.py` →
`conductor/v0.py`/`hybrid.py` → `dmx/encode.py` → `eval/` и `sim/renderer.py`.

[`cli.py`](lightsem/cli.py) удобно читать для понимания раннего pipeline по
правилам; современные кампании запускаются скриптами из
[`scripts/`](scripts/README.md). Более подробный разбор архитектуры находится
в [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Данные, запуски и история экспериментов

The [festival scene](docs/FESTIVAL_SCENE.md) adds a complete Blender venue and a
configurable multi-universe DMX patch, offline music-video rendering, and a live
Art-Net receiver. It expands the accepted controller into arbitrary fixture
counts without changing the selected model or the existing diagnostic renderer.

| Path | Purpose |
| --- | --- |
| `lightsem/model/light_layers/` | September 22 model, targets, training, evaluation, and overlay algorithms |
| `lightsem/audio/`, `conductor/`, `dmx/`, `motion/`, `sim/` | Reusable audio-to-light pipeline |
| `lightsem/experiments/` | Earlier model studies, grouped by campaign |
| `scripts/` | Maintained launchers, data preparation, diagnostics, and vault tooling |
| `experiments/<campaign>/records/` | Protocols, reports, receipts, and preserved research history |
| `experiments/<campaign>/outputs/` | Existing repository-local results and deliveries |
| `runs/` | New transient local outputs; ignored by Git |
| `/mnt/lightsem-data/` | Datasets, caches, checkpoints, and large experiment artifacts; existing paths are unchanged |

Use `bash run.sh <script>` for the CPU environment and `bash run_torch.sh <script>` for the dedicated Torch environment. Launchers now live under `scripts/`; see [the script index](scripts/README.md). The research [campaign index](experiments/README.md) and [relocation map](experiments/RELOCATION_MAP.json) locate historical files by their old path.

The historical records keep their original contents and absolute path claims. Those paths are documentary after this move; use the relocation map to find the files. Replaying a frozen experiment from its original binding requires a new, versioned binding for the new source paths. No new training, scoring, or model promotion occurred in this reorganization.

[Проверка C на другой музыке](experiments/2026-09-30-cross-song/README.md): два новых аудиовхода, диагностика одинаковых движений и залипания палитры, два ролика для просмотра. Проверки до рендера закреплены в [AGENTS.md](AGENTS.md).
