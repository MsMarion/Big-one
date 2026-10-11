import sys
from PIL import Image
im = Image.open(sys.argv[1])
n = im.n_frames
for k in [int(x) for x in sys.argv[2].split(",")]:
    im.seek(max(0, min(n - 1, k if k >= 0 else n + k)))
    im.convert("RGB").save(f"/repo/data/gif_frame_{k}.png")
print(n, "frames")
