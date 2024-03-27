import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np

def calc_hist_normalized(img):
    histSize = 256
    histRange = (0, 256)
    # calculate the histograms
    b_hist = cv2.calcHist([img], [0], None, [histSize], histRange) / (img.shape[0] * img.shape[1])
    g_hist = cv2.calcHist([img], [1], None, [histSize], histRange) / (img.shape[0] * img.shape[1])
    r_hist = cv2.calcHist([img], [2], None, [histSize], histRange) / (img.shape[0] * img.shape[1])

    return b_hist, g_hist, r_hist

# add histogram and image to the same plot
def show_image_with_hist(img, title="Image"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # normal colors to display
    b_hist, g_hist, r_hist = calc_hist_normalized(img) # calculate the histograms
    cumulative_hist_b,cumulative_hist_g, cumulative_hist_r = np.cumsum(b_hist), np.cumsum(g_hist), np.cumsum(r_hist) # calculate the cumulative histograms
    
    gs = plt.GridSpec(2, 4, width_ratios=[3, 1, 1, 1])

    plt.figure(figsize=(13, 5))

    plt.suptitle(title, fontsize=16)
    ax0 = plt.subplot(gs[:, 0]) # for image 
    ax0.set_title('Image')
    ax1 = plt.subplot(gs[0, 1:4]) # for rgb hist
    ax1.set_title('RGB Histogram')

    ax2 = plt.subplot(gs[1, 1:3])
    ax2.set_title('Cumulative RGB Histogram')
    ax3 = plt.subplot(gs[1, 3])
    ax3.set_title('Average Histogram')

    # set the x axis limits for the histogram subplots and grid
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([0, 256])
        ax.grid(True)

    # display the image
    ax0.imshow(img_rgb)
    ax0.axis('off')

    # all 3 histograms
    ax1.plot(b_hist, color='b')
    ax1.plot(g_hist, color='g')
    ax1.plot(r_hist, color='r')

    # cumulative histograms
    ax2.plot(cumulative_hist_b, color='b')
    ax2.plot(cumulative_hist_g, color='g')
    ax2.plot(cumulative_hist_r, color='r')

    # add 3 colors hist
    avg_hits = (b_hist + g_hist + r_hist) / 3
    ax3.hist(np.arange(0, 256), bins=256, weights=avg_hits, color='black')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.subplots_adjust(left=0.05, right=0.95)

    # save image
    plt.savefig(f"results/{title}.png")


def linear_leveling_transformation(img):
    # split the image into its 3 channels
    b, g, r = cv2.split(img)

    # calculate the histograms
    hist = calc_hist_normalized(img)

    # calculate the cumulative histograms
    cumulative_histogram_b = np.cumsum(hist[0]) 
    cumulative_histogram_g = np.cumsum(hist[1])
    cumulative_histogram_r = np.cumsum(hist[2])

    # apply the transformation
    b = np.clip(255 * cumulative_histogram_b[b], 0, 255)
    g = np.clip(255 * cumulative_histogram_g[g], 0, 255)
    r = np.clip(255 * cumulative_histogram_r[r], 0, 255)

    # merge the channels back
    return cv2.merge([b, g, r]).astype(np.uint8)


def arithmetic_transformation(img, b_delta, g_delta, r_delta):
    # split the image into its 3 channels
    b, g, r = cv2.split(img)

    # add the delta to each channel and divide by 255
    b = np.clip((b + b_delta), 0, 255) 
    g = np.clip((g + g_delta), 0, 255) 
    r = np.clip((r + r_delta), 0, 255) 

    # merge the channels back
    return cv2.merge([b, g, r])


def dynamic_range_expansion_transformation(img, alpha):
    # find maximum and minimum values for each channel

    # if the image is of type uint8, convert it to float64
    if img.dtype == 'uint8':
        img_converted = img.astype(np.float64) / 255
    
    b, g, r = cv2.split(img_converted)
    b_min, b_max = b.min(), b.max()
    g_min, g_max = g.min(), g.max()
    r_min, r_max = r.min(), r.max()

    # apply the transformation
    b = np.clip(((b - b_min) / (b_max - b_min)) ** alpha, 0, 1)
    g = np.clip(((g - g_min) / (g_max - g_min)) ** alpha, 0, 1)
    r = np.clip(((r - r_min) / (r_max - r_min)) ** alpha, 0, 1)

    img_transformed = cv2.merge([b, g, r])  # merge the channels back
    
    # convert the image back to uint8 if it was initially of that type
    if img.dtype == 'uint8':
        img_transformed = (255 * img_transformed).clip(0, 255).astype(np.uint8)

    return img_transformed


def uniform_transformation(img):
    b, g, r = cv2.split(img)
    # calculate the histograms
    hist = calc_hist_normalized(img)

    b_min, b_max = b.min(), b.max()
    g_min, g_max = g.min(), g.max()
    r_min, r_max = r.min(), r.max()

    cumulative_histogram_b = np.cumsum(hist[0]) 
    cumulative_histogram_g = np.cumsum(hist[1])
    cumulative_histogram_r = np.cumsum(hist[2])

    b = np.clip((b_max - b_min) * cumulative_histogram_b[b] + b_min, 0, 255)
    g = np.clip((g_max - g_min) * cumulative_histogram_g[g] + g_min, 0, 255)
    r = np.clip((r_max - r_min) * cumulative_histogram_r[r] + r_min, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)


def exponential_transformation(img, alpha):
    b, g, r = cv2.split(img)
    # calculate the histograms
    hist = calc_hist_normalized(img)

    b_min, b_max = b.min(), b.max()
    g_min, g_max = g.min(), g.max()
    r_min, r_max = r.min(), r.max()

    cumulative_histogram_b = np.cumsum(hist[0]) 
    cumulative_histogram_g = np.cumsum(hist[1])
    cumulative_histogram_r = np.cumsum(hist[2])

    b = b_min - 255/alpha * np.log(1 - cumulative_histogram_b[b]) 
    g = g_min - 255/alpha * np.log(1 - cumulative_histogram_g[g]) 
    r = r_min - 255/alpha * np.log(1 - cumulative_histogram_r[r]) 

    return (cv2.merge([b, g, r])).astype(np.uint8).clip(0, 255)


def rayleigh_law_transformation(img, alpha):
    b, g, r = cv2.split(img)
    # calculate the histograms
    hist = calc_hist_normalized(img)

    b_min, b_max = b.min(), b.max()
    g_min, g_max = g.min(), g.max()
    r_min, r_max = r.min(), r.max()

    cumulative_histogram_b = np.cumsum(hist[0]) 
    cumulative_histogram_g = np.cumsum(hist[1])
    cumulative_histogram_r = np.cumsum(hist[2])

    b = np.clip(b_min + (2*alpha**2 * np.log(1 / (1 - cumulative_histogram_b[b]))) ** 0.5 * 255, 0, 255)
    g = np.clip(g_min + (2*alpha**2 * np.log(1 / (1 - cumulative_histogram_g[g]))) ** 0.5 * 255, 0, 255)
    r = np.clip(r_min + (2*alpha**2 * np.log(1 / (1 - cumulative_histogram_r[r]))) ** 0.5 * 255, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)


def two_thirds_rule_transformation(img):
    b, g, r = cv2.split(img)
    # calculate the histograms
    hist = calc_hist_normalized(img)

    b_min, b_max = b.min(), b.max()
    g_min, g_max = g.min(), g.max()
    r_min, r_max = r.min(), r.max()

    cumulative_histogram_b = np.cumsum(hist[0]) 
    cumulative_histogram_g = np.cumsum(hist[1])
    cumulative_histogram_r = np.cumsum(hist[2])

    b = np.clip((cumulative_histogram_b[b] ** (2/3) * 255), 0, 255)
    g = np.clip((cumulative_histogram_g[g] ** (2/3) * 255), 0, 255)
    r = np.clip((cumulative_histogram_r[r] ** (2/3) * 255), 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)


def hyperbolic_transformation(img, alpha):
    b, g, r = cv2.split(img)
    # calculate the histograms
    hist = calc_hist_normalized(img)

    b_min, b_max = b.min(), b.max()
    g_min, g_max = g.min(), g.max()
    r_min, r_max = r.min(), r.max()

    cumulative_histogram_b = np.cumsum(hist[0]) 
    cumulative_histogram_g = np.cumsum(hist[1])
    cumulative_histogram_r = np.cumsum(hist[2])

    b = np.clip(alpha ** cumulative_histogram_b[b] * 255, 0, 255)
    g = np.clip(alpha ** cumulative_histogram_g[g] * 255, 0, 255)
    r = np.clip(alpha ** cumulative_histogram_r[r] * 255, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)


def LUT_transformation(img, LUT):
    return cv2.LUT(img, LUT).astype(np.uint8)


def show_image_profile(img, level, title="Profile"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # normal colors to display
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(img_rgb)
    plt.axis('off') # disable the axis

    plt.subplot(1, 2, 2)
    profile = img[level, :]
    plt.plot(profile)
    plt.title('Profile')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.subplots_adjust(left=0.05, right=0.95)

    # save image
    plt.savefig(f"results/{title}.png")


def show_image_projection(img, title="Projection"):
    b, g, r = cv2.split(img)
    # calculate 0x projection
    projection_x = (np.sum(b, axis=0) + np.sum(g, axis=0) + np.sum(r, axis=0)) / img.shape[0] / 3

    # calculate 0y projection
    projection_y = (np.sum(b, axis=1) + np.sum(g, axis=1) + np.sum(r, axis=1)) / img.shape[1] / 3

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.grid(True)
    plt.title('Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 3)
    plt.ylim([255, 0])
    plt.xlim([0, img.shape[1]])
    plt.grid(True)
    plt.plot(range(img.shape[1]), projection_x, color='g')
    plt.title('Projection X')

    plt.subplot(2, 2, 2)
    plt.xlim([0, 255])
    plt.ylim([img.shape[1], 0])
    plt.grid(True)
    plt.plot(projection_y, range(img.shape[0]), color='g')
    plt.title('Projection Y')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.subplots_adjust(left=0.05, right=0.95)

    # save image
    plt.savefig(f"results/{title}.png")


# open the image
img = cv2.imread("img.jpeg")
assert img is not None, "File could not be read"

# show the image and its histogram
show_image_with_hist(img, 'Source image')

#linear leveling transformation
linear_leveling_transformed_img = linear_leveling_transformation(img)
show_image_with_hist(linear_leveling_transformed_img, 'Linear leveling transformation')

# arithmetics transformation
arithmetic_transformed_img = arithmetic_transformation(img, 10, 20, 30)
show_image_with_hist(arithmetic_transformed_img, 'Arithmetics transformation')

# dynamic range expansion
dynamic_range_expansion_img = dynamic_range_expansion_transformation(img, 0.8)
show_image_with_hist(dynamic_range_expansion_img, 'Dynamic range expansion')

# uniform transformation
uniform_transformed_img = uniform_transformation(img)
show_image_with_hist(uniform_transformed_img, 'Uniform transformation')

# expositional transformation
expositional_transformed_img = exponential_transformation(img, 4)
show_image_with_hist(expositional_transformed_img, 'Expositional transformation')

# rayleigh law transformation
rayleigh_law_transformed_img = rayleigh_law_transformation(img, 0.5)
show_image_with_hist(rayleigh_law_transformed_img, 'Rayleigh law transformation')

# two thirds rule transformation
two_thirds_rule_transformed_img = two_thirds_rule_transformation(img)
show_image_with_hist(two_thirds_rule_transformed_img, 'Two thirds rule transformation')

# hyperbolic transformation
hyperbolic_transformed_img = hyperbolic_transformation(img, 0.1)
show_image_with_hist(hyperbolic_transformed_img, 'Hyperbolic transformation')

# LUT transformation
# generate LUT 
lut = np.arange(256, dtype = np.uint8)
lut = np.clip(np.power(lut, 0.9) + 20, 0, 255)

LUT_transformed_img = LUT_transformation(img, lut)
show_image_with_hist(LUT_transformed_img, 'LUT transformation')

### image profile

# open the image
img = cv2.imread("barcode.jpg")
assert img is not None, "File could not be read"
show_image_profile(img, img.shape[0] // 2)

### image projection

# open the image
img = cv2.imread("fill.jpg")
assert img is not None, "File could not be read"

show_image_projection(img, "Projection 1")

img = cv2.imread("fill2.jpg")
assert img is not None, "File could not be read"

show_image_projection(img, "Projection 2")

plt.show()