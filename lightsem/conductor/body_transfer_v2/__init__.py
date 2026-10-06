"""Paper-inspired, source-dependent body transfer research portfolio."""
from .common import prepare_features
from .installation import make_outputs, adapt_field


CATALOG = {
    '01': ('Bone directions', ['1804.05653']),
    '02': ('Joint bends and planes', ['1804.05653']),
    '03': ('Articulated light sculpture', ['2005.05732']),
    '04': ('Pose-aware anatomical pooling', ['2306.08006']),
    '05': ('Support-relative transfer', ['2109.07431']),
    '06': ('Relational distance constellation', ['2303.08658']),
    '07': ('Near/far interaction field', ['2410.20986']),
    '08': ('Proximity bridges', ['2109.07431']),
    '09': ('Differential-coordinate field', ['2509.26633']),
    '10': ('Coherent body deformation', ['0905.2635']),
    '11': ('Balanced anatomical transport', ['1306.0895']),
    '12': ('Graph-spectral deformation', ['1211.0053']),
    '13': ('Anatomical harmonic field', ['1211.0053']),
    '14': ('Bearing-preserving formation', ['1408.6552']),
    '15': ('Laban shape and effort', ['2006.06071', '2509.24469']),
}


def algorithms():
    from .anatomical import FUNCTIONS as anatomical
    from .relational import FUNCTIONS as relational
    from .composition import FUNCTIONS as composition
    return {str(key).zfill(2): value for key, value in
            {**anatomical, **relational, **composition}.items()}


def map_body(body, outputs, algorithm_id, fps=40.):
    features = prepare_features(body, fps)
    field = algorithms()[str(algorithm_id).zfill(2)](features).validate(len(body))
    result = adapt_field(features, field, outputs)
    result['algorithm_id'] = str(algorithm_id).zfill(2)
    return result
