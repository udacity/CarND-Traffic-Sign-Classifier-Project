import cv2
import numpy as np
import matplotlib.pyplot as plt

from lane_line import LaneLine

# Matplotlib figure size to have large enough image plotting
FIGURE_SIZE = (12, 6)

def plot_bgr(img):
    """ A helper for plotting a BGR image with matplotlib """
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
def plot_gray(gray):
    """ A helper for plotting a grayscale image with matplotlib """
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(gray, cmap='gray')

def plot_roi(img, roi):
    """ A helper for plotting a 4-point ROI on a given image """
    if len(img.shape) == 2:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        output = img.copy()
    roi_color = (255, 0, 255)
    thickness = 4
    cv2.polylines(output, [roi], True, roi_color, thickness)
    return output

def get_plottable_curves(height, left_fit, right_fit, xm_per_pix, ym_per_pix, steps=20):
    ploty = np.linspace(0, height, steps)
    left_curve = np.int32(list(zip(
        LaneLine.evaluate_poly2(left_fit, ploty * ym_per_pix) / xm_per_pix, ploty)))
    right_curve = np.int32(list(zip(
        LaneLine.evaluate_poly2(right_fit, ploty * ym_per_pix) / xm_per_pix, ploty)))
    return left_curve, right_curve
    
def plot_lane_curves(lane_img, left_curve, right_curve, thickness=6):
    out_img = cv2.polylines(lane_img, [left_curve], False, [0,255,255], thickness)
    out_img = cv2.polylines(out_img, [right_curve], False, [255,255,0], thickness)
    return out_img

def plot_lane_poly_on(lane_img, left_curve, right_curve, color=[100,200,100]):
    """
    Plots a polygon, representing a found road lane on a bird-eye image.
    Polygon is formed by two lane curves, left and right.
    [!] Modifies the given image.
    """
    # Note: right curve's points must go in the opposite order to maintain a
    # polygon's points traversing order.
    points = np.append(left_curve, right_curve[::-1], axis=0)
    cv2.fillPoly(lane_img, [points], color)
    
def plot_radiuses_on(img, l_radius, r_radius, color=(20, 20, 20), fontScale = 1.3, thickness = 3):
    """
    Plots the lane curvature radiuses as textstring at the top of the image.
    [!] Modifies the given image.
    """
    height, width = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # As the curvature radius calculation is not precise, I round
    # results to 10 meters (e.g. to the second integer digit)
    message = "Lane radius: ~{:4.0f} m    (left {:4.0f} m, right {:4.0f} m)".format(
                                round((l_radius + r_radius) / 2, -1),
                                round(l_radius, -1),
                                round(r_radius, -1), )

    origin = (int(width * 0.05), int(height * 0.1)) 
    cv2.putText(img, message, origin, font, fontScale, (200,200,200), 15, cv2.LINE_AA)
    cv2.putText(img, message, origin, font, fontScale, color, thickness, cv2.LINE_AA)

def plot_hcenter_offset_on(img, loffset, roffset, thickness = 1):
    """
    Expects two offset values from left and right lane respectively.
    The offset value is negative, if the center of the car is shifted to the right
    side of the screen (image), and positive if the car center is shifted to the
    left side of the screen (image). The offset values are in millimeters.
    Plots the offset value as textstring at the bottom of the image.
    [!] Modifies the given image.
    """
    offset = loffset + roffset
    height, width = img.shape[:2]
    green = (50, 150, 50)
    blue = (150, 50, 50)
    message = "Offset {:4.0f} mm" .format(round(abs(offset), -1))
    r_origin = (int(width * 0.52), int(height * 0.9)) 
    l_origin = (int(width * 0.20), int(height * 0.9)) 
    origin = r_origin if (offset > 0) else l_origin
    color = green if (offset > 0) else blue
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1.2; thickness = 2
    cv2.putText(img, message, origin, font, fontScale, (200,200,200), 10, cv2.LINE_AA)
    cv2.putText(img, message, origin, font, fontScale, color, thickness, cv2.LINE_AA)

def get_lane_template_birdeye(shape, lane_width = 60):
    """
    Returns an image with two white vertical stripes
    that might be used as a highcontrast lane lines template image
    """
    width = shape[1]
    left_line_desired_position = width * 1 // 4
    right_line_desired_position = width * 3 // 4
    template = np.zeros(shape, dtype=np.uint8)
    template[:, left_line_desired_position-lane_width:left_line_desired_position+lane_width] = 255
    template[:, right_line_desired_position-lane_width:right_line_desired_position+lane_width] = 255
    return template