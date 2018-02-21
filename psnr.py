import math

def calculate_psnr(original_y, encoded_y, original_cb, encoded_cb, original_cr, encoded_cr, npix):
    error_y = 0  
    error_cb = 0  
    error_cr = 0
    for i in range(0, len(original_y)):
        dif_y = abs(original_y[i] - encoded_y[i])  
        dif_cb = abs(original_cb[i] - encoded_cb[i])  
        dif_cr = abs(original_cr[i] - encoded_cr[i])  
        error_y += dif_y * dif_y  
        error_cb += dif_cb * dif_cb  
        error_cr += dif_cr * dif_cr  

    mse_y = float(error_y) / float(npix)  
    mse_cb = float(error_cb) / float(npix)  
    mse_cr = float(error_cr) / float(npix)  

    if mse_y != 0:
        psnr_y = float(-10.0 * math.log(mse_y / (255 * 255), 10))
    else:
        psnr_y = 0
    if mse_cb != 0:
        psnr_cb = float(-10.0 * math.log(mse_cb / (255 * 255), 10))
    else:
        psnr_cb = 0
    if mse_cr != 0:
        psnr_cr = float(-10.0 * math.log(mse_cr / (255 * 255), 10))
    else:
        psnr_cr = 0

    print("PSNR={0}".format(psnr_y))