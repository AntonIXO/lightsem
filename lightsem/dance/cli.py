"""Native dance and emitted-light CLI commands; fixed C is the default."""
from pathlib import Path
import json
import time
from .baseline import DEFAULT_CONFIG


def add_options(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--wav', type=Path, help='Original complete song')
    source.add_argument('--aist', help='AIST music ID from the local waveform collection')
    parser.add_argument('--baseline', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--d4max', '--d4-max', action='store_true', help='Enable own-song MuQ-indexed motif recomposition')
    parser.add_argument('--d6', action='store_true', help='Enable external learned positive pacing; D4 is optional')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='Native body/planner device; default reproduces the verified FP32 CPU path')
    parser.add_argument('--checkpoint', type=Path, help='Alternate location of the exact pinned fixed-C weights')
    parser.add_argument('--d6-checkpoint', type=Path, help='Alternate location of the pinned D6 weights')
    parser.add_argument('--features', type=Path, help='Authenticated own-song frozen MuQ MusicTimeline cache')
    parser.add_argument('--base92', type=Path, help='Authenticated Base92 cache paired with --features')
    parser.add_argument('--audio-python', help='Interpreter containing pinned MuQ/source/structure dependencies')
    parser.add_argument('--audio-device', choices=('cpu', 'cuda'), help='Device for uncached audio/P2 extraction')
    parser.add_argument('--audio-cache', type=Path, help='Whole-song feature cache mount; default /mnt/lightsem-data/cache/native_c_cli')
    parser.add_argument('--seed', type=int, default=None, help='Generation draw seed; default7001, checkpoint remains seed0')


def _audio(args):
    if args.wav:
        return args.wav.expanduser().resolve()
    from .baseline import REPO
    return REPO / 'data/wav/wav' / (args.aist + '.wav')


def _generate(args, root):
    from .pipeline import generate
    return generate(_audio(args), root, baseline_config=args.baseline, d4max=args.d4max, d6=args.d6,
                    device=args.device, checkpoint=args.checkpoint, d6_checkpoint=args.d6_checkpoint,
                    timeline_path=args.features, base92_path=args.base92, draw_seed=args.seed,
                    audio_python=args.audio_python, audio_device=args.audio_device, audio_cache=args.audio_cache)


def do_dance(args):
    from .baseline import REPO
    root = (args.out or REPO / 'runs/dance' / str(time.time_ns())).expanduser().resolve()
    *_, report = _generate(args, root)
    print(json.dumps({'run': str(root), 'body': str(root / 'body.npz'),
                      'baseline': report['baseline_id'], 'MuQ': True,
                      'd4max': args.d4max, 'd6': args.d6, 'status': report['status']}, indent=2))
    return 0


def do_render(args):
    from .baseline import REPO
    from .lighting import map_lights
    from .render import render_video
    output = (args.out or REPO / 'runs/dance' / (str(time.time_ns()) + '.mp4')).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    root = output.with_name(output.stem + '_run')
    body, _, _, config, generation = _generate(args, root)
    if generation['status'] in {'failure_diagnostic', 'unqualified'}:
        raise RuntimeError(f"Refusing to render generation with status {generation['status']}; inspect {root / 'GENERATION.json'}")
    counts = {key: getattr(args, key) if getattr(args, key) is not None else value
              for key, value in config['lighting']['counts'].items()}
    lighting = map_lights(body, _audio(args), root, config, counts=counts, color_path=args.colors,
                         audio_python=args.audio_python, audio_device=args.audio_device)
    render_video(body, root, _audio(args), output, lighting, start=args.start, duration=args.duration)
    print(json.dumps({'video': str(output), 'run': str(root), 'baseline': generation['baseline_id'],
                      'MuQ': True, 'd4max': args.d4max, 'd6': args.d6,
                      'fixture_motion_limits': False, 'status': generation['status']}, indent=2))
    return 0
