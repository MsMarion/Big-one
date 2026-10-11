"""User photo ingest: validate with PIL, apply EXIF rotation, strip metadata (incl. GPS), downscale,
write a JPEG plus a thumbnail under data_dir/photos/<object_id>/."""
import io
import os
import secrets

from PIL import Image, ImageOps

try:  # phone photos (iPhone HEIC)
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from . import config

MAX_UPLOAD = 25 * 1024 * 1024


def thumb_of(rel_path):
    root, _ = os.path.splitext(rel_path)
    return root + ".thumb.jpg"


def ingest(object_id, data: bytes):
    """Returns the data-relative path of the stored photo. Raises ValueError for non-images."""
    if len(data) > MAX_UPLOAD:
        raise ValueError("image larger than 25 MB")
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()  # structural check; must reopen afterwards
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        raise ValueError(f"not a readable image: {e}") from None
    p = config.load().get("photos", {})
    rel = f"photos/{object_id}/{secrets.token_hex(4)}.jpg"
    full = os.path.join(config.data_dir(), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    big = im.copy()
    big.thumbnail((p.get("max_side", 1024),) * 2)
    big.save(full, "JPEG", quality=90)  # re-encode: drops EXIF/GPS
    small = im.copy()
    small.thumbnail((p.get("thumb_side", 320),) * 2)
    small.save(os.path.join(config.data_dir(), thumb_of(rel)), "JPEG", quality=85)
    return rel
