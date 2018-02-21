from PIL import Image

def get_image_data(filename):
    im = Image.open(filename)
    width = im.size[0]
    height = im.size[1]
    npix = im.size[0] * im.size[1]
    return width, height, npix

def get_rgb(filename, npix):    
    im = Image.open(filename)
    rgb_im = im.convert('RGB')
    r = [-1] * npix
    g = [-1] * npix
    b = [-1] * npix
    for y in range(0, im.size[1]):
        for x in range(0, im.size[0]):
            rpix, gpix, bpix = rgb_im.getpixel((x, y))
            r[im.size[0] * y + x] = rpix
            g[im.size[0] * y + x] = gpix
            b[im.size[0] * y + x] = bpix
    return r, g, b

def get_yuv(filename):
    im = Image.open(filename)
    im = im.convert('YCbCr')
    y = []
    u = []
    v = []
    for pix in list(im.getdata()):
        y.append(pix[0])
        u.append(pix[1])
        v.append(pix[2])
    return y, u, v

def rgb_to_yuv(r, g, b):  
    y = [0] * len(r)
    cb = [0] * len(r)
    cr = [0] * len(r)
    for i in range(0, len(r)):
        y[i] = int(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i])
        cb[i] = int(128 - 0.168736 * r[i] - 0.331364 * g[i] + 0.5 * b[i])
        cr[i] = int(128 + 0.5 * r[i] - 0.418688 * g[i] - 0.081312 * b[i])
    return y, cb, cr