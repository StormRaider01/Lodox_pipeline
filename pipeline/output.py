"""Phase 6: final output.

Persists the bone-suppressed image and a summary report of the CNR/SNR
validation results computed in Phase 5 / Phase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import io as skio


def save_image_png(img: np.ndarray, path: str) -> None:
    """Save a float image as an 8-bit PNG, normalised to its own min/max
    (display-only; not a substitute for saving the original float data
    alongside it if quantitative reuse is needed).
    """
    rng = img.max() - img.min()
    normalized = (img - img.min()) / rng if rng > 0 else np.zeros_like(img)
    skio.imsave(path, (normalized * 255).astype(np.uint8), check_contrast=False)


def build_summary_report(results: dict) -> pd.DataFrame:
    """Flatten a nested {stage: {metric: value}} results dict into a tidy
    DataFrame for display/CSV export.
    """
    rows = []
    for stage, stage_metrics in results.items():
        if isinstance(stage_metrics, dict) and all(isinstance(v, dict) for v in stage_metrics.values()):
            for sub_stage, sub_metrics in stage_metrics.items():
                rows.append({"stage": f"{stage}.{sub_stage}", **sub_metrics})
        else:
            rows.append({"stage": stage, **stage_metrics})
    return pd.DataFrame(rows)


def save_report_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
