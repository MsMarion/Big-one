from PIL import Image
im = Image.open("/repo/data/jaw_held_sdf.png")
im.crop((250, 150, 500, 330)).resize((500, 360), Image.LANCZOS).save("/repo/data/jaw_held_sdf_zoom.png")
print("ok")
