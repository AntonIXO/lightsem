"""Six paper-inspired, deterministic body-to-installation deformation fields.

These are small CPU adaptations, not reproductions of learned paper models.
All operators use the fixed canonical 22-joint geometry; neither the song nor
the requested fixture count changes a basis, normalizer, or correspondence.
The common readout expands the 22 samples to physical fixtures. Coordinates
are deformations in metres, with zero corresponding to the canonical pose.
No temporal integration, independent clock, random source, or retiming occurs.

References and the precise scope of each adaptation are recorded below and
in every FieldResult.meta. NumPy is the only numerical dependency.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .common import (
    BODY_PARTS,
    REFERENCE,
    ROLE_ORDER,
    Features,
    FieldResult,
    all_attribution,
    derivative,
    skeleton_laplacian,
)
from .skeleton import PARENTS


def _meta(identifier, name, paper, equation, constants, limitations):
    return {
        "algorithm_id": identifier,
        "name": name,
        "paper": paper,
        "adaptation": equation,
        "constants": constants,
        "limitations": limitations,
        "spatial_domain": "fixed 22-node virtual installation",
        "temporal_source": "body pose and explicitly stated body derivatives only",
    }


@lru_cache(maxsize=1)
def _probes():
    """Fixed volumetric query sites, close to corresponding canonical joints.

    The depth offsets sample the deformation *off* the nearly planar skeleton.
    Offsets belong to anatomy, never to frame/time or fixture count.
    """
    probes = REFERENCE.copy()
    for part, depth in zip(BODY_PARTS, (0.10, 0.22, -0.22, -0.14, 0.14)):
        probes[list(part), 2] += depth
    probes.setflags(write=False)
    return probes


def _gaussian(a, b, width):
    squared_distance = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
    return np.exp(-squared_distance / (2.0 * width * width))


@lru_cache(maxsize=1)
def _cpd_operator():
    kernel = _gaussian(REFERENCE, REFERENCE, 0.40)
    query_kernel = _gaussian(_probes(), REFERENCE, 0.40)
    # K_query (K + ridge I)^-1, computed with a solve, not explicit inversion.
    result = np.linalg.solve(kernel + 0.12 * np.eye(22), query_kernel.T).T
    result.setflags(write=False)
    return result


def coherent_gaussian_body_deformation(features: Features) -> FieldResult:
    """ID 10: known-correspondence Gaussian coherent deformation.

    Inspired by CPD, Myronenko & Song, arXiv:0905.2635, nonrigid Gaussian
    displacement field / linear-system equations 19 and 22. Here anatomical
    correspondences are GIVEN: W=(K+0.12I)^-1(P-P_ref), field=K_query W.
    No EM, correspondence inference, fitted bandwidth, or registration is run.
    """
    field = np.einsum("ij,tjc->tic", _cpd_operator(), features.displacement)
    return FieldResult(field, all_attribution(), _meta(
        10, "coherent_gaussian_body_deformation", "https://arxiv.org/abs/0905.2635",
        "W=(K+lambda I)^-1(P-P_ref); D(Q)=K(Q,P_ref)W; fixed known correspondences",
        {"gaussian_width_m": 0.40, "ridge": 0.12,
         "part_query_depth_m": [0.10, 0.22, -0.22, -0.14, 0.14]},
        ["Gaussian smoothing may suppress localized articulation.",
         "Signed interpolation weights can overshoot between source joints.",
         "This adapts the CPD kernel form, not its EM registration algorithm."],
    ))


@lru_cache(maxsize=1)
def _transport_operator():
    """Balanced, fixed source-to-query transport with uniform capacities."""
    labels = np.zeros(22, dtype=int)
    for label, part in enumerate(BODY_PARTS):
        labels[list(part)] = label
    spatial_cost = np.sum((_probes()[:, None] - REFERENCE[None, :]) ** 2, axis=-1) / 0.60**2
    anatomical_cost = 1.20 * (labels[:, None] != labels[None, :])
    kernel = np.exp(-(spatial_cost + anatomical_cost) / 0.23)
    marginal = np.full(22, 1.0 / 22)
    u = np.ones(22)
    v = np.ones(22)
    for _ in range(10000):
        u = marginal / (kernel @ v)
        v = marginal / (kernel.T @ u)
        plan = u[:, None] * kernel * v[None, :]
        if np.max(np.abs(plan.sum(axis=1) - marginal)) < 1e-12:
            break
    else:
        raise RuntimeError("Static anatomical Sinkhorn plan did not converge")
    # The row-normalized matrix transfers deformation; its fixed marginals
    # MUST NOT be used as lamp intensity (that would impose constant light).
    operator = plan / plan.sum(axis=1, keepdims=True)
    operator.setflags(write=False)
    plan.setflags(write=False)
    return operator, plan


def fixed_balanced_sinkhorn_transport(features: Features) -> FieldResult:
    """ID 11: Cuturi 2013, arXiv:1306.0895, Section 4 / equations 2-3.

    Entropic transport P=diag(u) exp(-C/epsilon) diag(v), fixed once on
    canonical geometry. Uniform row/column mass makes anatomical coverage
    explicit. Only the body deformation is time-dependent.
    """
    operator, plan = _transport_operator()
    field = np.einsum("ij,tjc->tic", operator, features.displacement)
    meta = _meta(
        11, "fixed_balanced_sinkhorn_transport", "https://arxiv.org/abs/1306.0895",
        "T=diag(u)exp(-C/epsilon)diag(v); field=row_normalize(T)(P-P_ref)",
        {"spatial_scale_m": 0.60, "anatomical_mismatch_cost": 1.20,
         "entropy_epsilon": 0.23, "max_iterations": 10000,
         "row_marginal_tolerance": 1e-12, "marginals": "uniform 1/22"},
        ["Entropic averaging can cancel opposing joint motions.",
         "Additional physical fixtures do not add source information.",
         "Transport capacities are not brightness values."],
    )
    meta["marginal_error"] = float(max(np.max(np.abs(plan.sum(0) - 1 / 22)),
                                        np.max(np.abs(plan.sum(1) - 1 / 22))))
    meta["mean_row_entropy_nats"] = float(-np.mean(np.sum(
        operator * np.log(np.maximum(operator, 1e-300)), axis=1)))
    return FieldResult(field, all_attribution(), meta)


def _canonical_eigenbasis(laplacian):
    """Fix eigenspace rotation and sign using canonical coordinate axes.

    Eigenvalue order alone does not fix degenerate eigenvectors. Projecting
    fixed node-order axes into each eigenspace and Gram-Schmidt orthogonalizing
    fixes that ambiguity without a frame-dependent alignment or random seed.
    """
    values, raw_basis = np.linalg.eigh(laplacian)
    basis = np.empty_like(raw_basis)
    first = 0
    while first < len(values):
        last = first + 1
        while last < len(values) and abs(values[last] - values[first]) < 1e-9:
            last += 1
        projector = raw_basis[:, first:last] @ raw_basis[:, first:last].T
        vectors = []
        for index in range(len(values)):
            vector = projector[:, index].copy()
            for prior in vectors:
                vector -= np.dot(prior, vector) * prior
            norm = np.linalg.norm(vector)
            if norm > 1e-9:
                vector /= norm
                # First significant entry is a deterministic sign anchor.
                pivot = np.flatnonzero(np.abs(vector) > 1e-9)[0]
                vector *= 1.0 if vector[pivot] >= 0 else -1.0
                vectors.append(vector)
            if len(vectors) == last - first:
                break
        if len(vectors) != last - first:
            raise RuntimeError("Could not construct deterministic graph eigenspace")
        basis[:, first:last] = np.stack(vectors, axis=1)
        first = last
    return values, basis


@lru_cache(maxsize=1)
def _spectral_operators():
    # A fixed path links semantic display slots in ROLE_ORDER. Its modes are
    # spatial patterns only; body graph coefficients provide all timing.
    adjacency = np.zeros((22, 22))
    for a, b in zip(ROLE_ORDER[:-1], ROLE_ORDER[1:]):
        adjacency[a, b] = adjacency[b, a] = 1.0
    output_laplacian = np.diag(adjacency.sum(1)) - adjacency
    _, body_basis = _canonical_eigenbasis(skeleton_laplacian())
    _, fixture_basis = _canonical_eigenbasis(output_laplacian)
    source = body_basis[:, :9]  # DC plus eight nonconstant spatial modes.
    target = fixture_basis[:, :9]
    operator = target @ source.T
    for value in (source, target, operator):
        value.setflags(write=False)
    return source, target, operator


def fixed_graph_spectrum_transfer(features: Features) -> FieldResult:
    """ID 12: fixed body-graph Fourier coefficients drive fixture-graph modes.

    Shuman et al., arXiv:1211.0053, Section II, graph Fourier transform and
    inverse equations 3-4. This is a hand-defined correspondence of ordered
    spatial modes, not a claim that their anatomical semantics are identical.
    """
    source, _, operator = _spectral_operators()
    field = np.einsum("ij,tjc->tic", operator, features.displacement)
    coefficients = np.einsum("jk,tjc->tkc", source, features.displacement)
    total_energy = float(np.sum(features.displacement**2))
    retained = float(np.sum(coefficients**2))
    meta = _meta(
        12, "fixed_graph_spectrum_transfer", "https://arxiv.org/abs/1211.0053",
        "C=U_body[:,0:9]^T(P-P_ref); field=U_virtual[:,0:9]C",
        {"nonconstant_modes": 8, "virtual_graph": "22-node ROLE_ORDER path",
         "degenerate_eigenvalue_tolerance": 1e-9,
         "basis_rule": "canonical-axis projections, Gram-Schmidt, first-positive sign"},
        ["Truncation loses high spatial frequency articulation.",
         "Ordered graph modes have no guaranteed cross-graph semantic equivalence.",
         "Global mixed attribution is intentional; this is not isolated limb control."],
    )
    meta["retained_source_displacement_energy_fraction"] = retained / total_energy if total_energy else 1.0
    return FieldResult(field, all_attribution(), meta)


@lru_cache(maxsize=1)
def _virtual_laplacian():
    squared = np.sum((_probes()[:, None] - _probes()[None, :])**2, axis=-1)
    adjacency = np.zeros((22, 22))
    for index in range(22):
        # Canonical index provides deterministic tie-breaking.
        candidates = np.lexsort((np.arange(22), squared[index]))
        neighbors = candidates[candidates != index][:3]
        adjacency[index, neighbors] = np.exp(-squared[index, neighbors] / (2 * 0.45**2))
    adjacency = np.maximum(adjacency, adjacency.T)
    # Canonical skeletal links guarantee connectedness without an N-dependent
    # nearest-neighbor graph or a per-frame change in correspondence.
    for joint, parent in enumerate(PARENTS[1:], 1):
        adjacency[joint, parent] = adjacency[parent, joint] = max(adjacency[joint, parent], 0.20)
    laplacian = np.diag(adjacency.sum(1)) - adjacency
    laplacian.setflags(write=False)
    return laplacian


@lru_cache(maxsize=1)
def _harmonic_operator():
    anchors = np.array([20, 21, 7, 8, 9, 15])
    interior = np.array([j for j in range(22) if j not in anchors])
    laplacian = _virtual_laplacian()
    operator = np.zeros((22, 22))
    # Extremity anchors also encode the adjacent bone's deformation, so an
    # elbow/knee movement with a fixed endpoint remains observable.
    anchor_sources = np.zeros((len(anchors), 22))
    for row, joint in enumerate(anchors):
        if row < 4:
            anchor_sources[row, joint] = 1.45
            anchor_sources[row, PARENTS[joint]] = -0.45
        else:
            anchor_sources[row, joint] = 1.0
    operator[anchors] = anchor_sources
    interpolation = np.linalg.solve(laplacian[np.ix_(interior, interior)],
                                    -laplacian[np.ix_(interior, anchors)])
    operator[interior] = interpolation @ anchor_sources
    attribution = [sorted({0, *np.flatnonzero(np.abs(row) > 1e-12).tolist()}) for row in operator]
    operator.setflags(write=False)
    return operator, attribution


def anatomically_anchored_harmonic_extension(features: Features) -> FieldResult:
    """ID 13: exact Dirichlet anchors and a spatial minimum-energy extension.

    Shuman et al., arXiv:1211.0053, Section II-E / equation 6, gives the graph
    Dirichlet energy d^T L d. Minimizing with fixed anchor values yields
    L_uu d_u=-L_ua d_a. This has no diffusion in *time* and no damping state.
    """
    operator, attribution = _harmonic_operator()
    field = np.einsum("ij,tjc->tic", operator, features.displacement)
    return FieldResult(field, [list(a) for a in attribution], _meta(
        13, "anatomically_anchored_harmonic_extension", "https://arxiv.org/abs/1211.0053",
        "argmin d^T L_virtual d with exact anatomical anchors; L_uu d_u=-L_ua d_a",
        {"anchor_joints": [20, 21, 7, 8, 9, 15], "distal_bone_gain": 0.45,
         "nearest_neighbors": 3, "edge_width_m": 0.45,
         "minimum_skeleton_edge_weight": 0.20},
        ["Unanchored fine articulation may be lost.",
         "Interior nodes interpolate anchors and contain no independent dance signal.",
         "Exact anchor differences can be attenuated later by fixture limits."],
    ))


_BEARING_EDGES = tuple((parent, joint) for joint, parent in enumerate(PARENTS[1:], 1)) + (
    (20, 21), (7, 8), (20, 8), (21, 7), (18, 19), (4, 5), (15, 20), (15, 21),
)


def _bearing_solve(positions, displacement):
    """Batched 66-variable positive-definite QP, bounded memory per chunk.

    q minimizes sum ||P_g(q_j-q_i)||^2 + lambda||q-q_data||^2.
    Positive lambda=0.30 anchors translation/scale even in non-rigid or
    degenerate bearing graphs. Near-coincident source edges are masked.
    """
    answers = np.empty_like(positions)
    identity = np.eye(3)
    for begin in range(0, len(positions), 128):
        end = min(begin + 128, len(positions))
        p = positions[begin:end]
        matrix = np.broadcast_to(0.30 * np.eye(66), (len(p), 66, 66)).copy()
        blocks = matrix.reshape(len(p), 22, 3, 22, 3)
        for a, b in _BEARING_EDGES:
            edge = p[:, b] - p[:, a]
            norm = np.linalg.norm(edge, axis=-1)
            bearing = edge / np.maximum(norm[:, None], 1e-6)
            projector = identity - bearing[:, :, None] * bearing[:, None, :]
            projector *= (norm > 1e-6)[:, None, None]
            blocks[:, a, :, a, :] += projector
            blocks[:, b, :, b, :] += projector
            blocks[:, a, :, b, :] -= projector
            blocks[:, b, :, a, :] -= projector
        targets = _probes()[None] + 0.50 * displacement[begin:end]
        rhs = (0.30 * targets).reshape(len(p), 66, 1)
        answers[begin:end] = np.linalg.solve(matrix, rhs).reshape(len(p), 22, 3)
    return answers


@lru_cache(maxsize=1)
def _bearing_reference():
    result = _bearing_solve(REFERENCE[None], np.zeros((1, 22, 3)))[0]
    result.setflags(write=False)
    return result


def bearing_preserving_formation(features: Features) -> FieldResult:
    """ID 14: instantaneous bearing-shape QP, not a dynamical controller.

    Zhao & Zelazo, arXiv:1408.6552, Section II, P_g=I-gg^T and bearing rigidity
    equations 4-7. We borrow the projector residual only. We do not assert the
    framework is bearing rigid or adopt their temporal stabilization law.
    """
    field = _bearing_solve(features.positions, features.displacement) - _bearing_reference()[None]
    return FieldResult(field, all_attribution(), _meta(
        14, "bearing_preserving_formation", "https://arxiv.org/abs/1408.6552",
        "argmin_Q sum_edges||(I-gg^T)(Q_j-Q_i)||^2+lambda||Q-Q_data||^2; subtract canonical solution",
        {"data_anchor_weight": 0.30, "data_displacement_gain": 0.50,
         "edge_count": len(_BEARING_EDGES), "minimum_source_edge_m": 1e-6,
         "solve_chunk_frames": 128},
        ["A soft projector penalty does not guarantee bearing rigidity.",
         "The positive data term anchors scale and translation and prevents zero collapse.",
         "Near-coincident body samples have no defined bearing; their edges are masked.",
         "Independent per-pose fitting still requires the common mechanical limiter."],
    ))


def laban_instantaneous_shape_effort(features: Features) -> FieldResult:
    """ID 15: instantaneous axis-aligned shape and kinetic-effort descriptors.

    Samadani, Gorbet & Kulic, arXiv:2006.06071, motivates Effort/Shape
    descriptors. We extract fixed-axis body spans, envelope volume and local
    kinetic energy only; no HMM, generated motion, PCA, or song normalization.
    Axis-aligned envelope volume is a coarse *proxy* for occupied body volume;
    energy/energy increase are kinematic proxies, not measured force or emotion.
    """
    p = features.positions
    spans = np.ptp(p, axis=1)
    reference_spans = np.ptp(REFERENCE, axis=0)
    span_change = spans - reference_spans
    # Fixed canonical scales avoid amplification by a song's small variation.
    spatial_scales = np.array([1.60, 1.80, 0.80])
    canonical_centered = REFERENCE - REFERENCE.mean(axis=0)
    spatial_shape = canonical_centered / spatial_scales
    volume = np.prod(spans, axis=1)
    reference_volume = float(np.prod(reference_spans))
    # Bounded metre-like expansion, with exactly zero at canonical volume.
    volume_change = 0.35 * np.tanh((volume - reference_volume) / 1.50)
    body_center_change = p.mean(axis=1) - REFERENCE.mean(axis=0)
    field = (0.20 * features.displacement
             + 0.65 * span_change[:, None, :] * spatial_shape[None]
             + volume_change[:, None, None] * spatial_shape[None]
             + 0.20 * body_center_change[:, None, :])
    energy = 0.5 * np.sum(features.velocity**2, axis=-1)
    increasing_energy = np.maximum(derivative(energy, features.fps), 0.0)
    activity = -np.expm1(-energy / 0.32 - increasing_energy / 2.0)
    meta = _meta(
        15, "laban_instantaneous_shape_effort", "https://arxiv.org/abs/2006.06071",
        "D=.20 displacement+.65 span_change*fixed_shape+.35 tanh(volume_change/1.5)*fixed_shape+.20 centroid_change; activity=1-exp(-E/.32-max(dE/dt,0)/2)",
        {"canonical_spatial_scales_m": spatial_scales.tolist(),
         "local_displacement_gain": 0.20, "span_gain": 0.65,
         "volume_gain_m": 0.35, "volume_scale_m3": 1.50,
         "centroid_gain": 0.20, "kinetic_energy_scale": 0.32,
         "positive_energy_derivative_scale": 2.0},
        ["Axis-aligned spans are sensitive to body orientation and extremal joints.",
         "Envelope volume is not mesh volume; kinetic effort is not force or emotion.",
         "Global shape descriptors discard fine articulation; the local term retains some.",
         "The first frame uses zero backward-difference velocity by common contract."],
    )
    return FieldResult(field, all_attribution(), meta, activity)


FUNCTIONS = {
    10: coherent_gaussian_body_deformation,
    11: fixed_balanced_sinkhorn_transport,
    12: fixed_graph_spectrum_transfer,
    13: anatomically_anchored_harmonic_extension,
    14: bearing_preserving_formation,
    15: laban_instantaneous_shape_effort,
}
