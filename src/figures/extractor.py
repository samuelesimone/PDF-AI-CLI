"""
FigureExtractor — extracts figure images from a PDF using PyMuPDF.

Two complementary strategies:
  Vector figures  — page.cluster_drawings() groups PDF drawing commands into
                    bounding boxes.  Most academic plots (matplotlib, pgfplots,
                    gnuplot) are pure-vector and never appear in get_images().
  Raster images   — page.get_images() surfaces embedded PNG/JPEG assets such
                    as photographs and scanned figures.

Raster images that are fully contained within an already-identified vector
region are skipped to avoid duplicate analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

# Minimum bounding-box area (PDF points²) to qualify as a real figure.
# 1 pt = 1/72 inch → 7200 pt² ≈ a 1.4 × 1.4 inch box, filtering tiny icons.
_MIN_AREA: int = 7200

# Minimum absolute dimension on each side (points).
_MIN_SIDE: int = 50

# Maximum fraction of the page area — skip backgrounds/full-page borders.
_MAX_PAGE_FRACTION: float = 0.85

# DPI used when rendering vector regions to PNG for the VLM.
_RENDER_DPI: int = 150

# Image formats accepted by Ollama's /api/chat vision endpoint.
_ACCEPTED_EXTS = frozenset({"png", "jpeg", "jpg"})


@dataclass
class ExtractedFigure:
    """A single figure extracted from a PDF page."""
    page: int           # 1-based page number
    index: int          # sequential index across all pages
    source: str         # "vector" | "raster"
    image_bytes: bytes  # raw image bytes (PNG for vector, PNG/JPEG for raster)
    width: int          # pixels
    height: int         # pixels


def extract_figures(pdf_path: Path) -> List[ExtractedFigure]:
    """Return every figure found in *pdf_path* as a list of :class:`ExtractedFigure`.

    Raises nothing — a page that fails to process is silently skipped so the
    rest of the document can still be indexed.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    results: List[ExtractedFigure] = []
    scale = _RENDER_DPI / 72.0
    mat = fitz.Matrix(scale, scale)
    global_idx = 0

    for page_num, page in enumerate(doc):
        page_area = page.rect.width * page.rect.height
        vector_rects: list = []

        # ── Vector figures (matplotlib, pgfplots, TikZ, …) ───────────────
        if hasattr(page, "cluster_drawings"):
            for raw_rect in page.cluster_drawings():
                r = fitz.Rect(raw_rect)
                if not _is_valid_region(r, page_area):
                    continue
                try:
                    pix = page.get_pixmap(matrix=mat, clip=r)
                    results.append(ExtractedFigure(
                        page=page_num + 1,
                        index=global_idx,
                        source="vector",
                        image_bytes=pix.tobytes("png"),
                        width=pix.width,
                        height=pix.height,
                    ))
                    vector_rects.append(r)
                    global_idx += 1
                except Exception:
                    continue

        # ── Raster images (photographs, scanned figures, …) ───────────────
        seen_xrefs: set = set()
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            rects_on_page = page.get_image_rects(xref)
            if not rects_on_page:
                continue
            img_rect = fitz.Rect(rects_on_page[0])

            if not _is_valid_region(img_rect, page_area):
                continue

            # Skip raster images already covered by a vector region
            if any(_mostly_contained(img_rect, vr) for vr in vector_rects):
                continue

            base = doc.extract_image(xref)
            if not base or not base.get("image"):
                continue

            ext = base.get("ext", "").lower().lstrip(".")
            if ext not in _ACCEPTED_EXTS:
                continue

            results.append(ExtractedFigure(
                page=page_num + 1,
                index=global_idx,
                source="raster",
                image_bytes=base["image"],
                width=int(img_rect.width),
                height=int(img_rect.height),
            ))
            global_idx += 1

    doc.close()
    return results


# ── Pure-geometry helpers (no fitz dependency) ────────────────────────────────


def _is_valid_region(rect, page_area: float) -> bool:
    """Return True if *rect* is large enough to be a meaningful figure."""
    w, h = rect.width, rect.height
    if w < _MIN_SIDE or h < _MIN_SIDE:
        return False
    area = w * h
    if area < _MIN_AREA:
        return False
    if page_area > 0 and area / page_area > _MAX_PAGE_FRACTION:
        return False
    return True


def _mostly_contained(inner, outer, threshold: float = 0.80) -> bool:
    """Return True if *inner* rect is ≥ *threshold* covered by *outer*."""
    ix0 = max(inner.x0, outer.x0)
    iy0 = max(inner.y0, outer.y0)
    ix1 = min(inner.x1, outer.x1)
    iy1 = min(inner.y1, outer.y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return False
    inner_area = inner.width * inner.height
    if inner_area <= 0:
        return False
    return ((ix1 - ix0) * (iy1 - iy0)) / inner_area >= threshold
