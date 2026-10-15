"""
RSNA Knee Abnormality Detection — Iteration 3 Research Loader
MEDICALLY CORRECT 3D DICOM VOLUME LOADER:
  1. True 3D slice depth sorting: projects ImagePositionPatient onto slice normal (r x c)
     with robust fallback to InstanceNumber.
  2. Aspect-preserving letterbox padding to square (prevents joint distortion).
  3. Global per-series 1st-99th percentile contrast windowing (preserves fluid/edema signal).
  4. PhotometricInterpretation handling (inverts MONOCHROME1 to MONOCHROME2).
"""
import os
import cv2
import numpy as np
import pydicom

def sort_series_slices(series_path):
    """
    Sorts DICOM slice files along their true physical anatomical axis
    using the slice normal vector n = r x c, where r and c are the
    row and column direction cosines from ImageOrientationPatient.
    """
    files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')]
    if not files:
        return []

    slice_positions = []
    valid_files = []

    for fpath in files:
        try:
            # Read header only without loading raw pixels (ultra fast)
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            if hasattr(dcm, 'ImagePositionPatient') and hasattr(dcm, 'ImageOrientationPatient'):
                iop = np.array(dcm.ImageOrientationPatient, dtype=float)
                row_dir = iop[:3]
                col_dir = iop[3:]
                slice_normal = np.cross(row_dir, col_dir)
                norm_len = np.linalg.norm(slice_normal)
                if norm_len > 1e-6:
                    slice_normal = slice_normal / norm_len
                ipp = np.array(dcm.ImagePositionPatient, dtype=float)
                pos = float(np.dot(ipp, slice_normal))
            elif hasattr(dcm, 'InstanceNumber'):
                pos = float(dcm.InstanceNumber)
            else:
                pos = 0.0
            slice_positions.append(pos)
            valid_files.append(fpath)
        except Exception:
            continue

    if not valid_files:
        return files

    sorted_indices = np.argsort(slice_positions)
    return [valid_files[idx] for idx in sorted_indices]


def pad_to_square(img):
    """Aspect-preserving letterbox padding to square (no squashing/distortion)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[pad_h: pad_h + h, pad_w: pad_w + w] = img
    return padded


def load_series_volume(series_path, n_slices=7, target_size=224):
    """
    Loads n_slices equidistant slices along the true anatomical z-depth,
    applies per-series 1st-99th percentile windowing, pads to square,
    and returns uint8 array of shape (n_slices, target_size, target_size).
    """
    blank = np.zeros((n_slices, target_size, target_size), dtype=np.uint8)
    if not os.path.exists(series_path):
        return blank

    sorted_files = sort_series_slices(series_path)
    if not sorted_files:
        return blank

    indices = np.linspace(0, len(sorted_files) - 1, n_slices, dtype=int)
    raw_slices = []

    for idx in indices:
        try:
            dcm = pydicom.dcmread(sorted_files[idx], stop_before_pixels=False)
            arr = dcm.pixel_array.astype(np.float32)

            # Handle RescaleSlope and RescaleIntercept if present
            slope = getattr(dcm, 'RescaleSlope', 1.0)
            intercept = getattr(dcm, 'RescaleIntercept', 0.0)
            arr = arr * slope + intercept

            # Handle MONOCHROME1 (minimum is white, invert to standard MONOCHROME2)
            if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
                arr = np.max(arr) - arr

            raw_slices.append(arr)
        except Exception:
            raw_slices.append(np.zeros((target_size, target_size), dtype=np.float32))

    # Global per-series 1st-99th percentile windowing (preserves fluid/edema signal!)
    all_pixels = np.concatenate([s.flatten() for s in raw_slices])
    p1 = np.percentile(all_pixels, 1.0)
    p99 = np.percentile(all_pixels, 99.0)
    if p99 <= p1:
        p99 = p1 + 1.0

    processed_slices = []
    for s in raw_slices:
        # Clip to per-series percentile window and scale to [0, 255]
        s_norm = np.clip((s - p1) / (p99 - p1), 0.0, 1.0) * 255.0
        s_u8 = s_norm.astype(np.uint8)

        # Aspect-preserving letterbox pad to square
        s_square = pad_to_square(s_u8)

        # Bilinear resize to target dimension
        s_resized = cv2.resize(s_square, (target_size, target_size), interpolation=cv2.INTER_AREA)
        processed_slices.append(s_resized)

    return np.stack(processed_slices, axis=0)  # (n_slices, target_size, target_size) uint8
