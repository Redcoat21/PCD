import cv2
import numpy as np
import matplotlib.pyplot as pyplot

def show_image(image):
    if image is not None:
        pyplot.imshow(image, cmap='gray')
        pyplot.axis('off')
        pyplot.show()
    else:
        print("the image is none")

def compare_image(image1, image2):
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(image1, cmap='gray')
    pyplot.title('original image')
    pyplot.axis('off')
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(image2, cmap='gray')
    pyplot.title('spatial filtered image')
    pyplot.axis('off')
    pyplot.show()

def compare_all(list_image):
    row = int(len(list_image) / 3)
    column = len(list_image) % 3

    if column != 0:
        row += 1

    for i in range(len(list_image)):
        pyplot.subplot(row, 3, i + 1)
        pyplot.imshow(list_image[i]["image"], cmap='gray')
        pyplot.title(list_image[i]["name"])
        pyplot.axis('off')

    pyplot.show()
        

def filtering(image, new_image, filter, weight):
    height = new_image.shape[0]
    width = new_image.shape[1]

    for i in range(1, height-1):
        for j in range(1, width-1):
            new_pxl = 0
            new_pxl += (image[i-1, j-1] * filter[0][0])
            new_pxl += (image[i-1, j] * filter[0][1])
            new_pxl += (image[i-1, j+1] * filter[0][2])
            new_pxl += (image[i, j-1] * filter[1][0])
            new_pxl += (image[i, j] * filter[1][1])
            new_pxl += (image[i, j+1] * filter[1][2])
            new_pxl += (image[i+1, j-1] * filter[2][0])
            new_pxl += (image[i+1, j] * filter[2][1])
            new_pxl += (image[i+1, j+1] * filter[2][2])
            new_pxl *= 1/weight
            if(new_pxl < 0):
                new_pxl = 0
            if(new_pxl > 255):
                new_pxl = 255
            new_image[i, j] = new_pxl

    return new_image

def lowpass_filtering(image):
    new_image = np.copy(image)
    filter = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    weight = 9
    new_image = filtering(image, new_image, filter, weight)

    return new_image

def highpass_filtering(image):
    new_image = np.copy(image)
    filter = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]   
    weight = 1
    new_image = filtering(image, new_image, filter, weight)

    return new_image

def hbf(image):
    new_image = np.copy(image)
    A = 1.5
    filter = [[-1, -1, -1], [-1, 8+A, -1], [-1, -1, -1]]   
    weight = 1
    new_image = filtering(image, new_image, filter, weight)

    return new_image

def prewitt1(image):
    new_image = np.copy(image)
    filter = [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]   
    weight = 1
    new_image = filtering(image, new_image, filter, weight)

    return new_image

def prewitt2(image):
    new_image = np.copy(image)
    filter = [[5, -3, -3], [5, 0, -3], [5, -3, -3]]   
    weight = 1
    new_image = filtering(image, new_image, filter, weight)

    return new_image

def sobel_filter1(image):
    new_image = np.copy(image)
    filter = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]   
    weight = 1
    new_image = filtering(image, new_image, filter, weight)

    return new_image

def sobel_filter2(image):
    new_image = np.copy(image)
    filter = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]   
    weight = 1
    new_image = filtering(image, new_image, filter, weight)

    return new_image

def median_blur(image):
    new_image = cv2.median_blur(image, 9)
    return new_image


image_path = "cat.jpg"
list_image = []
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(image.shape)
list_image.append({
    "name": "original image",
    "image": image
})

lpf_image = lowpass_filtering(image)
list_image.append({
    "name": "low pass filter image",
    "image": lpf_image
})

hpf_image = highpass_filtering(image)

list_image.append({
    "name": "high pass filter image",
    "image": hpf_image
})
hbf_image = hbf(image)

list_image.append({
    "name": "high boost filter image",
    "image": hbf_image
})
prewitt1_image = prewitt1(image)

list_image.append({
    "name": "prewitt 1 filter image",
    "image": prewitt1_image
})
prewitt2_image = prewitt2(image)

list_image.append({
    "name": "prewitt 2 filter image",
    "image": prewitt2_image
})
sobel1_image = sobel_filter1(image)

list_image.append({
    "name": "sobel 1 filter image",
    "image": sobel1_image
})
sobel2_image = sobel_filter2(image)

list_image.append({
    "name": "sobel 2 filter image",
    "image": sobel2_image
})
mb_image = median_blur(image)

list_image.append({
    "name": "median fitler filter image",
    "image": mb_image
})

compare_all(list_image)