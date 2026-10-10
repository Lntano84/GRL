# Artifact sealing correction

The first postrun review passed its step-cost, simulation-count, calibration and frozen-screen calculations, then rejected the delivery hash for outputs/grid05/analysis.log. The analysis generator had hashed its redirected stdout file before its final print and process closure. No experimental data, simulation or policy defect was found.

Retain the original manifest as delivery_manifest.INVALID_active_log.json. Exclude the active analysis stdout file and the review output from the generator manifest, regenerate the stable artifacts, then rerun only the saved-data arithmetic and hash review. No simulation, neural inference, training or main audit rerun is needed. Final postrun review records whether this passed.
