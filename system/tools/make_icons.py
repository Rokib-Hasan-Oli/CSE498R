"""Draw the mobile app icons (webapp/static/icon-192.png, icon-512.png): a white number plate on flag green."""
import os

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'webapp', 'static')
for n in (192, 512):
    im = Image.new('RGB', (n, n), (0, 106, 78))
    d = ImageDraw.Draw(im)
    s = n / 512
    d.rounded_rectangle([70 * s, 150 * s, 442 * s, 362 * s], radius=24 * s, fill=(251, 251, 247), outline=(17, 17, 17),
                        width=max(2, int(10 * s)))
    d.rectangle([110 * s, 190 * s, 402 * s, 230 * s], fill=(0, 106, 78))
    try:
        f = ImageFont.truetype('C:/Windows/Fonts/Nirmala.ttc', int(96 * s), index=0)
        d.text((n / 2, 300 * s), '১২-৩৪', font=f, fill=(17, 17, 17), anchor='mm')   # digits only: no conjunct shaping
    except OSError:
        d.rectangle([110 * s, 270 * s, 402 * s, 320 * s], fill=(17, 17, 17))
    d.ellipse([400 * s, 90 * s, 470 * s, 160 * s], fill=(211, 36, 58))
    im.save(os.path.join(OUT, f'icon-{n}.png'))
print('icons written')
