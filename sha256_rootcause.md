# SHA-256 Determinism Analysis Report

## Known Non-Determinism Sources (point-sprite path)
- NONE: current path uses no atomicAdd, no cross-warp reduction.
- Shader: shaders/point.vert:14-18 - per-vertex transform only; no shared or global accumulation.
- Shader: shaders/point.frag:6-14 - single outColor write at line 13, no parallel accumulation, no atomicAdd, no imageStore.
- Determinism guarantee: valid ONLY for point-sprite pipeline.

## Expected Non-Determinism Sources (future 3DGS path)
- shaders/alpha_composite.comp: line N/A (planned, not yet implemented)
  Risk: atomicAdd on RGB accumulator across warps
  -> floatNonAssociativity, see IEEE 754 sec. 5.9
- shaders/radix_sort.comp: line N/A (planned, not yet implemented)
  Risk: tie-breaking undefined for equal depth keys
  -> GaussianData.id must be used as tiebreaker

## Measurement result
- colorConsistency: 100.00% (100/100 frames identical)
- depthConsistency: 100.00% (100/100 frames identical)
- colorDivergedFrames: none
- depthDivergedFrames: none
- Conclusion: pass - point-sprite path is bit-stable for the measured tensors.

Rerun required after: v1.2 (alpha blending), v2.0 (GPU radix sort)
