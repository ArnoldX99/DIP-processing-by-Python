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

### Transfer picture into Grayscale and Bluescale
A grayscale image is an image in which each pixel is represented by a single value that corresponds to its brightness or intensity level. In other words, a grayscale image contains no color information, and the only variation in the image is the difference in brightness or darkness of each pixel.

Be different from RGB image which constructed by 3 channels, Grayscale images are typically represented using a single channel, where each pixel is represented by a single 8-bit value ranging from 0 to 255. A value of 0 represents black, while a value of 255 represents white. Values in between represent varying shades of gray.  What's more, we can transfer the image into Blue-color style, just like characters in Avatar.

```python
# Change it into gray figure   ** See the NOTE!!
gray_mat = [0.299,0.857,0.114]
gray_img = np.dot(img,gray_mat)
fig = plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.imshow(gray_img, cmap='gray')


# Change image color to Blue-style
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Transfer RGB image into HSV figure
blue_tint = np.zeros_like(hsv_img)
# Set the hue channel of the numpy array to a value that corresponds to blue. The hue channel in the HSV color space ranges from 0 to 179,
# with blue corresponding to a hue value of around 120:
blue_tint[:, :, 0] = 120
# Set the saturation and value channels of the numpy array to the corresponding channels of the HSV image:
blue_tint[:, :, 1] = hsv_img[:, :, 1]
blue_tint[:, :, 2] = hsv_img[:, :, 2]
# Transform it back to RGB image
blue_img = cv2.cvtColor(blue_tint, cv2.COLOR_HSV2RGB)
plt.subplot(1,2,2)
plt.imshow(blue_img)
```
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/20e777ec-1e25-4de4-8589-e8988b209075)

The operation is quite simpe for gray-scale figure-- we only need to multiply a $1 \times 3$ matrix. In fact you can just consider it as a 'weighted average' which can leads a RGB image to gray-scale.

**Note**: The formula of Gray-dgree processing：

$$Gray=R \times 0.299+G \times0.587+B \times0.114$$

Or we can directly use package: *gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)*, directly reach the goal.

## 2.Gray-Level Histogram

We've learnt in the previous chapter about gray-scale image. Now we want to do an important and useful visualization of it —— the gray-level histogram. A gray-level histogram is a graphical representation of the distribution of pixel intensities in a grayscale image. It shows the frequency of occurrence of each possible pixel intensity value in the image.

In a gray-level histogram, the x-axis represents the possible intensity levels (ranging from 0 to 255 for an 8-bit grayscale image), while the y-axis represents the number of pixels in the image that have that intensity level. The histogram can be used to analyze the overall brightness and contrast of an image, as well as the distribution of features and texture.

We can see the result of Gray-scale image of the girl's face ( You can see the code in the Jupyer notbook if you are interested):
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/bbaaf90e-3a73-423d-97f5-407555222c62)

From the histogram, we can see: the count of pixels mainly gathered in the interval of [0,150], which Indicate that the overall image has a dark color tone. Also, we can see a large number of points are concentrated around two gray levels 25 and 70. Bt observation of the image, we can guess that these two gray levels represent large areas of similar color tones in the image, such as hair or background.

### Linear Point Operations
Since we've knwon well about the Histogram of digital image, that is actually a 'numerical expression'. That means, we could apply some numerical or mathematical methods on it to try some handling. 

Fisrt of all, we are quite familiar with linear function we learnt in the high school: $y = kx + m$, where k and m are constant. Now the unkown $x$ is the gray-scale level of pixels.  Now we can make a rough guess and use code to verify whether it is reasonable: obviously, k can adjust the intensity of pixel gray values, and if it is negative, it means the inversion of brightness and darkness; m is used to increase the gray values of all pixels by a corresponding unit value, making the image darker or brighter overall.

Now let's try by code:
```python
from PIL import Image

# Open the image file
image = Image.open("map.jfif")

# Define the point operation function
def point_operation(pixel,k,m):
    # Extract the red, green, and blue values of the pixel
    r, g, b = pixel

    # Modify the pixel values using the point operation
    r_new = int(r*k+m)
    g_new = int(g*k+m)
    b_new = int(b*k+m)

    # Return the modified pixel values as a tuple
    return (r_new, g_new, b_new)

# Apply the point operation to each pixel in the image and we set k=1,b=-30
new_data_1 = [point_operation(pixel,1,-30) for pixel in image.getdata()]

# Create a new image with the modified pixel data
new_image_1 = Image.new(image.mode, image.size)
new_image_1.putdata(new_data_1)


# Apply the point operation to each pixel in the image and we set k=1.5,b=0
new_data_2 = [point_operation(pixel,1.5,0) for pixel in image.getdata()]
new_image_2 = Image.new(image.mode, image.size)
new_image_2.putdata(new_data_2)

# Apply the point operation to each pixel in the image and we set k=1,b=-50
new_data_3 = [point_operation(pixel,0.5,-20) for pixel in image.getdata()]
new_image_3 = Image.new(image.mode, image.size)
new_image_3.putdata(new_data_3)


# Save the modified image
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
ax[0][0].imshow(image)
ax[0][0].set_title('Original Image')
ax[0][1].imshow(new_image_1)
ax[0][1].set_title('Modified image k=1, b=-30')
ax[1][0].imshow(new_image_2)
ax[1][0].set_title('Modified image k=1.5, b=0')
ax[1][1].imshow(new_image_3)
ax[1][1].set_title('Modified image k=0.5, b=-20')
```
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/59176213-4bab-4a53-ad1c-ea03a9588fb5)

We can see by our operation for $y = k*x+b$: \
By setting $b = -30$, the map get darker and we can see the buliding clearer. \
By setting $k = 1.5$, we can see the contrast of map is larger. \
By setting $k = 0$ and $b = -20$, we can see the map get darker and contrast is smaller.

## Nonlinear Monotonic Point Operations
Unlike linear point operations, which are limited to simple linear functions, nonlinear point operations can use more complex functions that allow for more flexible and precise adjustments to the image. Nonlinear point operations can be used to enhance the visual quality of an image, such as improving the visibility of details in a low-contrast image or enhancing the color saturation of an image.

### Histogram Equalization
Histogram equalization is a technique used in image processing to improve the contrast of an image. The goal of histogram equalization is to stretch the intensity range of an image to better utilize the full range of pixel intensities. This can improve the visual quality of an image and make it easier to see details that may be difficult to discern in a low-contrast image.

![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/add0192b-4146-4c75-9e00-df22b34c213a)

In fact you can see some mathematical proof in Wikipedia if you want:

![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/63a7dbd2-5c69-4d56-acf0-f08d648e7a90)

Do remember that we need to do a round operation since we want integers, a popular way is :
$$y^{\prime}=round(y \cdot(L-1))$$

Let's see the example of operating image of moon's surface from satelite:

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load input image
img = cv2.imread('moon.jpeg', 0)

# Calculate histogram
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Calculate cumulative distribution function
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Perform histogram equalization
img_equalized = cv2.equalizeHist(img)

# Display input, output and their histograms
plt.figure(figsize=(10, 8))
plt.subplot(2,2,1), plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.subplot(2,2,2), plt.hist(img.flatten(), 256, [0, 256],color = 'gray')
plt.title('Input Histogram')
plt.subplot(2,2,3), plt.imshow(img_equalized, cmap='gray')
plt.title('Equalized Image')
plt.subplot(2,2,4), plt.hist(img_equalized.flatten(), 256, [0, 256], color = 'gray')
plt.title('Equalized Histogram')
plt.show()
```
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/1bf0a839-f11c-464f-aad8-d9a9fbeb3545)

We can see clearly there is a enhancement of image. We can see the detail clearly due to the raise of contrast. We can see the Histogram, the length of bright/dark area increase, and the 200-250 part raise from none to around 2000. That shows this method is usually used to increase the overall contrast of many images, especially when the contrast of useful data in the image is relatively similar. With this method, brightness can be better distributed on the histogram. This can be used to enhance local contrast without affecting global contrast, and histogram equalization achieves this by effectively expanding commonly used brightness values.

Now we get a basic knowledge of Digital image processing. With this, we can further do more things useful like:
**Words detection and recognition**:
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/ab31da99-4f70-476e-a35f-d4c7a9ba75ae)
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/3e4520d1-b936-4df7-9167-9baff50df4ac)

**Immage recovering**:
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/ec8602fa-988b-46ea-aaac-21c147c60698)

**Imgage searching**:
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/be37cc25-d590-43c7-9bd9-2abfdd192df5)
![image](https://github.com/ArnoldX99/DIP-processing-by-Python/assets/64125777/cd02882d-2e1c-4494-8728-f67167bf3dcc)

You can see all of them as notebook in files. Further more, we will update with some machine learnning and deep learning methods.

