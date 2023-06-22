# DIP-processing-by-Python
## 1. Read digital image in python and some basic handling
### A brief introduction of some basic pakages
![图片1](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/36195b49-9d96-49da-af40-e3e2c414605f)
When dealing with digital image missions, cv2, PIL, and matplotlib are all useful Python librariesfor image processing, but they have different functionalities and purposes.

**cv2 (OpenCV)** is an advanced computer vision library used for real-time computer vision applications. It is primarily used for image and video processing, including tasks such as object detection, face recognition, and image segmentation. It is optimized for speed and efficiency and is often used in applications such as robotics, surveillance, and self-driving cars.

**PIL (Python Imaging Library)** is a library used for image processing tasks such as image resizing, cropping, and color adjustments. It provides a wide range of image file format support and is often used for web development and scientific applications.

**Matplotlib** is a widely used library for data visualization, including the creation of 2D and 3D plots, scatter plots, histograms, and other types of charts. While it is not explicitly an image processing library, it can be used to display and manipulate images as well.

In conclusion, cv2 is focused on computer vision tasks, and it can handle many complex situations, PIL is focused on image processing tasks, and matplotlib is focused on data visualization tasks, but all three libraries can be used for image manipulation to some extent.

### How to read digital image in Python?
Let's start from the most basic operation: how to read the image? Well, let's try to read image of my favourite character  Mathilda in the film ***Léon*** by *matplotlib* and *cv2*:
```python
import cv2
import matplotlib.pyplot as plt

# Load the image by matplotlib
img = plt.imread('a.jpg')
plt.imshow(img)

# Load the image by cv2
img1 = cv2.imread('a.jpg')
plt.imshow(img1)
```
The result will be like following:
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/24e83e70-ab9d-4e64-b3c8-095ce62d6448)

Well I bet you can find something strange: there is something wrong with the figure read by cv2! That's a really important point I want to talk about. In fact you can use other method to read cv2 (you can easily search it online), but since I write the code by jupyter notebook, it can not display the figure since method provided by cv2 need to display the image in a new window. You can try it on python in your computer rather then notebook. Anyway, what I want to say is: 

**NOTICE**: We have different ways to load pictures in python, and their contruction are *different*! For example,plt.imread and PIL,Image.open read image data in the order of **RGB**，but cv2.imread read it in **BGR**! That leads the difference between two figures. To modify, we need to Convert the color channels from BGR to RGB:
```python
# Convert the color channels from BGR to RGB
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# Show the image using matplotlib
plt.imshow(img1)
plt.show()
```
You can find after that, you will get a normal beautiful face of the girl.

### Picture binarization by PILLOW
Okay, I believe you have a good command of how to read the image by python, whatever by Cv2 or matplotilb.(Or you can try PIL that I haven't got a try.) Suppose now we want to get a black and white figure like a old movie, that should be cool.

To say this procedure more academic, we call it 'Picture binarization'.Picture binarization is the process of converting a grayscale or color image into a **binary image**, where each pixel is either black or white (digitaly, 0 or 1). This is done by applying a threshold to the pixel values, such that any pixel value above the threshold is set to white, and any pixel value below the threshold is set to black. 
```python
# Change figure into black and white by pakage 'PLIIOW'
from PIL import Image
# Model 'L'is gray image，each pixel represented by 8 bits，0-black，255-white，number represents gray-scale.
img_PIL = Image.open('a.jpg')
Img = img_PIL.convert('L')

# Costumize the threshold of gray_scale，pixels larger than treshold change into black，conversely white.
threshold = 150

table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)

# Show the Binary image
bw_img = Img.point(table, '1')
plt.imshow(bw_img)

# HANDBOOK of PIL is https://pillow.readthedocs.io/en/latest/handbook/index.html
```

![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/ace719c9-70aa-41ac-b77c-4f8926bb01d9)

We can see the result: what a cool girl! I wish that could rise your interest in DIP and go furhter for learning it well.



