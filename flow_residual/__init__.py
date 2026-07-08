"""Flow-matching residual chunk model for boiling visualization.

Replaces the deterministic frame-AR regression in residual_video/ with:
- Stochastic flow-matching objective on residual chunks (T frames at once).
- Spatial nucleation prior conditioning.
- Audio cross-attention conditioning per chunk.
- Statistical (distribution-matching) evaluation.
"""
