"""
    Author: Jay Lal
    Common Utility functions/operations used in Computer Vision/Image Processing
    Dependencies: Python3, OpenCV>=3.4, Scikit-image>=0.15
"""

import cv2

from skimage.morphology import skeletonize as skel_zhang
from skimage.morphology import skeletonize_3d as skel_lee
from skimage.filters import threshold_sauvola, threshold_niblack, unsharp_mask
from skimage.measure import compare_psnr, compare_ssim
from skimage import transform as tf

import numpy as np
import math

from pathlib import Path

import os
import subprocess


# TODO: Create a resize function with dynamic interpolation (based on wether upscaling / downscaling is required
# Choose the best interpolation method, that is also the fastest (BICUBIC or AREA or LINEAR or whatever)

# glob pattern for fetching jpeg and png images
glob_img = "*[.png|.PNG][.jpg|.JPG]"


def _get_interpolation(inter_string):
    
    # In case it's already an interpolation code..
    if isinstance(inter_string, int):
        return inter_string
    
    inter_string = inter_string.lower().strip()
    inter_methods = {'area':cv2.INTER_AREA,
                     'linear':cv2.INTER_LINEAR,
                     'cubic':cv2.INTER_CUBIC,
                     'nearest':cv2.INTER_NEAREST}
    
    if inter_string not in inter_methods:
        raise Exception("Unknown Interpolation Method: '{}'".format(inter_string))
    
    return inter_methods[inter_string]


def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=[255, 255, 255],
                            flags=cv2.INTER_CUBIC)
    
#     rotated = scipy.ndimage.rotate(image, angle, reshape=False, cval=255)

    return rotated


def deskew_imgmagick(input_image):
    """
    :param input_image:
    :return: skewed_image
    uses imagemagick cmd_tool to correct skewness of the image
    """
    temp_directory = "."
    image_save_path = temp_directory + '/input_image.jpg'
    cv2.imwrite(image_save_path, input_image)
    file_to_save = temp_directory + '/deskwed_img.jpg'
    cmd = "convert " + image_save_path + " -grayscale average -deskew 40% " + file_to_save
    cmd_to_skew = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_out_pdf, cmd_err_pdf = cmd_to_skew.communicate()
    skew_image = cv2.imread(file_to_save)
    os.remove(image_save_path)
    os.remove(file_to_save)
    return skew_image


def ImageGenerator(img_collection, color='gray', target_size=None,
                   resize_max_h=None, resize_max_w=None,
                   preserve_ratio=False, inter='area'):
    """
        TODO: Can I make this ImageGenerator Fast / Make use of multithreading - for batches??
    """
    
    # If a directory is passed:
    if isinstance(img_collection, str) or not hasattr(img_collection, '__iter__'):
        img_collection = Path(img_collection).glob(glob_img)
    
    for img_path in img_collection:
        img = read_img(img_path, color)
        
        if target_size is not None:

            # Get the OpenCV interpolation code
            inter = _get_interpolation(inter)
            
            if isinstance(target_size, (int, float)):
                if preserve_ratio:
                    img = resize_padd_square(img, target_size, inter)
                else:
                    img = cv2.resize(img, (target_size, target_size), inter)
            
            # Target size is a tuple/list
            else:
                target_h, target_w = target_size
                if preserve_ratio:
                    raise NotImplementedError("Currently only square(same target height & width)\
resize and padd is supported to preserve aspect ratio")
                else:
                    img = cv2.resize(img, (target_h, target_w), inter)
            
        elif resize_max_w is not None or resize_max_h is not None:

            if resize_max_h and img.shape[0] > resize_max_h:
                img = resize(img, height=resize_max_h)
                
            if resize_max_w and img.shape[1] > resize_max_w:
                img = resize(img, width=resize_max_w)
            
        yield img_path, img

        
def resize_padd_square(img, desired_size, inter='area', padd_color=255):
    """
        TODO: Modify this function to resize_padd to non-square images as well..
    """
    
    if padd_color==255 and img.ndim == 3:
        padd_color = [255, 255, 255]
    
    inter = _get_interpolation(inter)
    
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    img = cv2.resize(img, (new_size[1], new_size[0]), inter)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=padd_color)

    return new_img


def symmetric_crop(img, h_crop_pct=0, w_crop_pct=0, copy=True):
    h,w = img.shape[:2]
    
    up_pct = h_crop_pct
    down_pct = 100 - up_pct
    
    left_pct = w_crop_pct
    right_pct = 100 - left_pct
    
    crop = img[int(h*up_pct/100): int(h*down_pct/100), int(w*left_pct/100): int(w*right_pct/100)].copy()
    
    if copy:
        crop = crop.copy()
    
    return crop
    

def remove_table_lines(img, hse=100, vse=40):
    # TODO: to_gray can be added as a decorator -> ensure_gray_img
    img = ~to_gray(img)

    # implt(img)
    # Specify size on horizontal axis
    horizontal_size = hse
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(img.copy(), horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    vertical_size = vse
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(img.copy(), verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # implt(horizontal)
    # implt(vertical)

    table_lines_mask = cv2.add(vertical, horizontal)
    thick_lines_mask = cv2.dilate(table_lines_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    #     thick_lines_mask = ~sharpen(~thick_lines_mask)
    # implt(thick_lines_mask)

    cleaned_img = cv2.add(~img, thick_lines_mask)
#     cleaned_img = cv2.medianBlur(cleaned_img, 5)
#     cleaned_img = ~cv2.morphologyEx(~cleaned_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
    # implt(cleaned_img)

    return cleaned_img


def sharpenOpenCV(image):

    image = to_gray(image)

    img_height = image.shape[0]
    radius, amount = int(math.ceil(img_height / 2.0)), int(math.ceil(img_height / 3.0))

    if radius % 2 == 0:
        radius = radius - 1

    # print("radius = ", radius)
    fimg = np.multiply(image, 1. / 255.0, dtype=np.float32)

    blurred = cv2.GaussianBlur(fimg, (radius, radius), 0)

    result = fimg + (fimg - blurred) * amount

    # clip the result between the given range...
    result = np.clip(result, 0.0, 1.0)

    sharp_img = (result * 255).astype(np.uint8)

    return sharp_img


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # (Borrowed from imutils)initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def img_extend(img, shape):
    """Extend 2D image (numpy array) in vertical and horizontal direction.
    Shape of result image will match 'shape'
    Args:
        img: image to be extended
        shape: shape (touple) of result image
    Returns:
        Extended image
    """
    x = np.zeros(shape, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    return x


def to_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

    
def to_bgr(img):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        
    return img


def sharpen(img, radius=5, amount=3, override=False):
    """
        Mathematically the following occurs..
        sharp_img = gray_img + amount * (gray_img - Gaussian(gray_img, kernel_raidus))
        Tweak the radius and amount according to image size
    """

    img_height = img.shape[0]
    if not override:
        print("overriding")
        radius, amount = int(math.ceil(img_height / 2.0)), int(math.ceil(img_height / 3.0))

    sharp_img = (unsharp_mask(to_gray(img), radius, amount) * 255).astype(np.uint8)
    return sharp_img


def threshold(img, method='otsu', blur_size=None, window_size=25, k=0.8):
    if blur_size:
        blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

    else:
        blur = img

    if method.strip().lower() == 'otsu':
        ret, thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if method.strip().lower() == 'sauvola':
        thresh_values = threshold_sauvola(blur, window_size=window_size)
        thresh_img = (blur > thresh_values).astype(np.uint8) * 255

    if method.strip().lower() == 'niblack':
        thresh_values = threshold_niblack(blur, window_size=window_size, k=k)
        thresh_img = (blur > thresh_values).astype(np.uint8) * 255

    return thresh_img


def skeletonize(binary_img, method='lee'):
    """Depreceated, use imutils.skeletonize, it's an opencv based implementation """
    # TODO: Add a check to see if the img passed is in-fact binary

    inv_binary_word_img = ~binary_img

    word_skeleton_img = None

    if method.strip().lower() == 'zhang':
        word_skeleton_img = skel_zhang(inv_binary_word_img / 255).astype(np.uint8) * 255

    elif method.strip().lower() == 'lee':
        word_skeleton_img = skel_lee(inv_binary_word_img / 255).astype(np.uint8)

    else:
        # Todo: raise exception with else unknown method
        pass

    return word_skeleton_img


def save_img(img_path, img):
    if not Path(img_path).parent.exists():
        print("Parent directory does not exist:", Path(img_path).parent)
        return False
    
    return cv2.imwrite(str(img_path), img)


def read_img(img_path, color='gray'):
    if not Path(img_path).exists():
        print("Image does not exist:", str(img_path))
        return False
    
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    color = color.lower().strip()
    color_reader = {'gray': to_gray(img),
                    'grey': to_gray(img),
                    'rgb': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    'bgr': img}

    if color not in color_reader:
        raise Exception("Unrecognised color format:'{}',\
        please use one of the valid_color formats:{}".format(color_reader.keys()))
        
    return color_reader[color]


def draw_bbox(img, boxes, color=(0, 255, 0), thickness=1):
    if len(boxes) != 0:
        if not isinstance(boxes[0], list):
            boxes = [boxes]
    else:
        return img

    bgr_img = to_bgr(img)
    boxed_img = bgr_img

    for box in boxes:
        x1, y1, x2, y2 = box
        boxed_img = cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, thickness)

    return boxed_img


def draw_bbox_with_text(img, bbox_with_text, bbox_color=(0, 255, 0), text_scale = 0.25, text_color=(0,0,255)):
    '''
        @prama:bbox_with_tex: list of dicts, each containing bbox & text
    '''
    assert len(bbox_with_text) > 0

    bgr_img = to_bgr(img)
    boxed_img = bgr_img

    for bbox in bbox_with_text:

        box = bbox['box']
        text = bbox['text']

        x1, y1, x2, y2 = box
        boxed_img = cv2.rectangle(bgr_img, (x1, y1), (x2, y2), bbox_color, 1)
        cv2.putText(boxed_img, text, (x1-3, y1-2), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, 1, cv2.LINE_AA)

    return boxed_img


def get_ssim_psnr(true_img, test_img, color=False):
    """Assuming input Images are color in BGR format (default in OpenCV)
    """
    
    def _get_luminance_BGR(bgr_img):
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    if not color:
        true_img = _get_luminance_BGR(true_img)
        test_img = _get_luminance_BGR(test_img)

    ssim = compare_ssim(true_img, test_img, multichannel=color)
    psnr = compare_psnr(true_img, test_img)
    
    return ssim, psnr


def get_pixel_density(region):
    # Assuming region is an inverted image..

    area = region.shape[0] * region.shape[1]
    pixel_intensity = region.sum()
    pixel_density = pixel_intensity / float(area)

    return pixel_density


def get_connected_component_stats(mask, order_reverse=True):
    '''
        @mask : inverted binary image (black background and white foreground)
        Returns Connceted Components: with additional column representing the label for that component
        this allows us to sort the components based on size, left/top co-ordinates, width, height, etc..
        without loosing their positon/label in the cca image..
    '''

    # Assert image is binary
    assert len(np.unique(mask)) <= 2

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask,
                                                                               connectivity=8)
    # Ignore the background component
    stats = stats[1:, :]
    new_stats = []

    for i, stat in enumerate(stats):
        new_stats.append(np.append(stat, i + 1))

    new_stats = sorted(new_stats, key=lambda k: k[cv2.CC_STAT_LEFT], reverse=order_reverse)
    new_stats = np.array(new_stats)

    return new_stats, output


def color_connected_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img


def get_tight_crop_bin(bin_image):
    # Works for binary images

    img_height, img_width = bin_image.shape[:2]

    upper_bound = 0
    for y1 in range(img_height):
        if 0 in bin_image[y1]:
            upper_bound = y1
            break

    lower_bound = img_height - 1
    for y2 in range(img_height - 1, -1, -1):
        if 0 in bin_image[y2]:
            lower_bound = y2
            break

    left_bound = 0
    for x1 in range(0, img_width):
        if 0 in bin_image.T[x1]:
            left_bound = x1
            break

    right_bound = img_width - 1
    for x2 in range(img_width - 1, -1, -1):
        if 0 in bin_image.T[x2]:
            right_bound = x2
            break

    return upper_bound, lower_bound, left_bound, right_bound


def get_tight_crop_gray(bin_image):
    # Works for grayscale images

    img_height, img_width = bin_image.shape[:2]

    upper_bound = 0
    for y1 in range(img_height):
        if np.any(bin_image[y1] != 255):
            upper_bound = y1
            break

    lower_bound = img_height - 1
    for y2 in range(img_height - 1, -1, -1):
        if np.any(bin_image[y2] != 255):
            lower_bound = y2
            break

    left_bound = 0
    for x1 in range(0, img_width):
        if np.any(bin_image[:, x1] != 255):
            left_bound = x1
            break

    right_bound = img_width - 1
    for x2 in range(img_width - 1, -1, -1):
        if np.any(bin_image[:, x2] != 255):
            right_bound = x2
            break

    return upper_bound, lower_bound, left_bound, right_bound


def create_shear(img, angle=math.pi / 4):
    # TODO: Hardcoded the padding size for now assuming the image to be of lower resolution
    # Can be calculated based on the input image size...
    padded_img = cv2.copyMakeBorder(img, top=50, bottom=150, left=50, right=150,
                                    borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

    cv2.imwrite('padded_img.jpg', padded_img)

    tform = tf.AffineTransform(shear=angle)

    sheared_img = (tf.warp(padded_img, tform, cval=1.0) * 255).astype(np.uint8)

    return sheared_img


def get_word_stroke_width(bin_word_img):
    # get black pixel count
    fg_pixel_sum = np.sum(bin_word_img == 0)

    # Skeletonize it
    skel_word_img = skeletonize(bin_word_img, method='lee')
    #     cv2.imwrite('skeletonzied_sauvola.jpg', skel_word_img)

    # get black pixel count
    skeleton_fg_pixel_sum = np.sum(skel_word_img == 255)

    # get stroke width
    # assert (skeleton_fg_pixel_sum != 0), "No text found in image"
    if skeleton_fg_pixel_sum == 0:
        print("****Warning: No text found in image!!!****")
        return 0

    stroke_width = fg_pixel_sum / skeleton_fg_pixel_sum

    return stroke_width


def draw_hough_lines(original_img, lines, write_output=False):
    # Ensure that the img is color image with 3 channels

    # The below for loop runs till r and theta values  
    # are in the range of the 2d array 

    img = original_img.copy()

    # if it's a grayscale image convert to color
    if len(img.shape) == 2:
        # print('Converting Grayscale image to color')
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    thetas = []

    for rtheta_pack in lines:

        r, theta = rtheta_pack[0][0], rtheta_pack[0][1]
        # Stores the value of cos(theta) in a 

        theta_degree = math.degrees(theta)

        # print(theta_degree)
        if (10 < theta_degree < 60) or (120 < theta_degree < 170):
            thetas.append(theta_degree)

            # print(theta_degree)
            a = np.cos(theta)

            # Stores the value of sin(theta) in b 
            b = np.sin(theta)

            # x0 stores the value rcos(theta) 
            x0 = a * r

            # y0 stores the value rsin(theta) 
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
            y2 = int(y0 - 1000 * (a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
            # (0,0,255) denotes the colour of the line to be  
            # drawn. In this case, it is red.
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Default value if no slant is detected
    theta_degree = 0
    if len(thetas) != 0:
        theta_degree = np.mean(thetas)

    if write_output:
        # print("shape of image:", img.shape)
        cv2.imwrite('hough_lines_output.png', img)

    return theta_degree


def correct_slant(word_img):
    # binarize it
    bin_word_img = threshold(word_img, method='sauvola', window_size=11)
    #     cv2.imwrite('bin_word_sauvola.jpg', bin_word_img)

    stroke_width = get_word_stroke_width(bin_word_img)
    #     print("Hadwritten stroke width:", stroke_width)

    ## Label all the connected components
    skel_word_img = skeletonize(bin_word_img, method='lee')

    # Calcluate length of L (vertical structuring element)
    # L = stroke_width / tan(theta)
    # Where theata is the maximum slant w.r.t vertical line
    # Maximum value of theta is 45 degrees, so tan(45) = 1, and L = stroke_width

    L = int(stroke_width)
    # Length of vertical structuring element
    # print("L:", L, end = ', ')

    ## Remove horizontal strokes, so that the remaining
    # vertical ones can be used to estimate slant angle
    vse = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    eroded_img = cv2.erode(~bin_word_img, kernel=vse, iterations=2)
    #     cv2.imwrite('eroded_img.jpg', eroded_img)
    dilated_img = cv2.dilate(eroded_img, kernel=vse)
    #     cv2.imwrite('dilated_img.jpg', dilated_img)

    opened_img = dilated_img

    inverse_open = ~opened_img
    #     cv2.imwrite('inverse_open.jpg', inverse_open)

    # Apply edge detection method on the image 
    edges = cv2.Canny(inverse_open, 50, 150, apertureSize=3)
    #     cv2.imwrite('canny_edge.jpg', edges)

    # This returns an array of r and theta values 
    lines = cv2.HoughLines(edges, 2, np.pi / 180, 50)

    if lines is None:
        # print("No lines detected!!!")
        #         cv2.imwrite(str(out_dir_unable / word_img_name), word_img)
        return word_img

    else:
        # print("Num of lines detected:", len(lines))

        #         theta_degrees = draw_hough_lines(word_img, lines, write_output=True)
        theta_degrees = draw_hough_lines(word_img, lines)
        # Average slant
        # print("Theta:", theta_degrees)

        sheared_img = create_shear(word_img, angle=math.radians(theta_degrees))
        #         cv2.imwrite('sheared.jpg', sheared_img)

        bin_sheared = threshold(sheared_img, method='sauvola', window_size=11)
        #         cv2.imwrite('bin_sheared.jpg', bin_sheared)

        y1,y2,x1,x2 = get_tight_crop_bin(bin_sheared)
        cropped_word = sheared_img[y1:y2, x1:x2]
        #         cv2.imwrite('sheared_cropped.jpg', cropped_word)

        return cropped_word


def viz_splits_final(img, totalSplits):
    for split in totalSplits:
        split = int(split)
        # ratio = imgSize[0] / imgSize[1]
        # split = int(split * ratio)
        cv2.line(img, (split, 0), (split, img.shape[0]), (0, 0, 255), 1)

    return img


def viz_splits_all(img, predictedSplits, confidence_threshold):
    predictedSplits = np.array(predictedSplits)
    # Get the index of splits having confidence greater than threshold!
    predictedSplits = np.where(predictedSplits >= confidence_threshold)[0]

    for split in predictedSplits:
        split = int(split)
        # ratio = imgSize[0] / imgSize[1]
        # split = int(split * ratio)
        cv2.line(img, (split, 0), (split, img.shape[0]), (0, 0, 255), 1)

    return img


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def improve_img_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(gray)

    h, w = img.shape
    dilate_kernel = (h // 150, w // 150)
    blurring_kernel = h // 50
    if blurring_kernel % 2 == 0:
        blurring_kernel = blurring_kernel + 1
    adjusted_ = adjust_gamma(img, 0.7)
    dilated_img = cv2.dilate(adjusted_, np.ones(dilate_kernel, np.uint8))
    bg_img = cv2.medianBlur(dilated_img, blurring_kernel)
    diff_img = 255 - cv2.absdiff(adjusted_, bg_img)
    norm_img = diff_img.copy()
    norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 225, 0, cv2.THRESH_TRUNC)
    norm_img = cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    norm_img = adjust_gamma(norm_img, 0.4)
    _, thr_img = cv2.threshold(norm_img, 255, 0, cv2.THRESH_TRUNC)

    return thr_img
