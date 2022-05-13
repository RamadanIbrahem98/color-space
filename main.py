import cv2
from color_space import RGB2LUV

image = cv2.imread('lena.jpg')
cv2_luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

LUV = RGB2LUV(rgb)

# For some reason the LUV values are not the same as the LUV values got from online websites for color space conversion.
# https://colormine.org/convert/rgb-to-luv
# and after mapping it to be from 0 to 255 space, it does not match that of the CV2 library.
# mapping is done following the formula in the link bellow.
# https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#MathJax-Element-56-Frame
print(image[:1, :1, :])
print(cv2_luv[:1, :1, :])
print(LUV[:1, :1, :])

cv2.imshow('LUV', LUV)
cv2.imshow('CV LUV', cv2_luv)
cv2.waitKey(0)
cv2.destroyAllWindows()
