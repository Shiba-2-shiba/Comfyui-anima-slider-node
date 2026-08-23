Upstream: https://github.com/muooon/QPOLA
Version: v1.0.4
Vendored on: 2026-08-23
Local modifications:
- Renamed upstream `optimizer/qpola.py` to `optimizer.py` for package-local import stability.
- Deferred CUDA driver and PTX loading until QPOLA is selected so CPU-only ComfyUI startup remains available.
- Restricted the integration path to CUDA-resident parameters; the upstream CPU-to-CUDA copy path is intentionally not exposed.
- Preserved the upstream CUDA source, PTX, and core parameter-update algorithm unchanged.
- Removed trailing whitespace and the extra final blank line from vendored kernel files; no executable content changed.
