# Capturing objects for objsearch

What makes identification and model matching work, roughly in order of impact.

1. **Metric depth + correct intrinsics.** Use the frames layout (`rgb/`, `depth/` uint16 mm, `cam_K.txt`).
   The measured L×W×H is the single strongest filter. An ESP-01 (24.8×14.3 mm) and a LEGO 2×4 (31.8×15.8×11.4 mm) can't be confused once the size is known.
2. **3–5 views per object.** Include one straight top-down view and the underside of PCBs: part numbers, pin labels and FCC IDs are often on one side only.
3. **A close-up of markings.** RF-shield text ("ESP-WROOM-32"), FCC IDs, chip markings, and the LEGO logo on the studs. OCR plus a part-number search usually settles the identity directly.
4. **Clean placement.** Use a matte background that contrasts with the objects, and leave ≥1 cm between objects so the depth-plane segmentation separates them. Avoid glossy or black-on-black where possible; depth sensors drop out on shiny surfaces.
5. **Optional scale check.** An ArUco/AprilTag board under the objects validates K and the depth scale.
6. **Optional hints.** A `hints.yaml` next to the capture (e.g. `{"0": "dev board", "notes": "all from the 2023 kit"}`) is passed to the agent.

For your own collection, add 1–3 phone photos per item on the dashboard (`/objects`, item page → drop photos):
top, bottom, markings close-up. Photos of your exact items are weighted above model renders during matching.
Uploads are re-encoded (EXIF/GPS stripped) and downscaled to 1024 px.

Free-tier note: images sent to the Gemini free tier may be used by Google to improve its products, and human reviewers may read them. Keep anything private out of the frame.
