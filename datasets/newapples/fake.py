from PIL import Image
import random
import os


def is_shade_of_red(r, g, b, threshold=50):
    """
    Determines if a pixel is a shade of red.
    A pixel is considered a shade of red if the red component is significantly greater than green and blue.
    """
    return r > g + threshold and r > b + threshold


def adjust_to_fruit_orange(r, g, b):
    """
    Adjusts the color to a hue resembling the fruit orange,
    with intensity proportional to the original red value.
    """
    scale = r / 255  # Scale intensity based on red component
    fruit_orange_r = int(255 * scale)
    # Deeper green component for richer orange
    fruit_orange_g = int(140 * scale)
    fruit_orange_b = int(0 * scale)    # Blue remains zero
    return (fruit_orange_r, fruit_orange_g, fruit_orange_b)


def adjust_to_ripe_orange(r, g, b, ran=120):
    """
    Adjusts the color to a vibrant, ripe orange hue,
    with intensity proportional to the original red value.
    """
    scale = r / 255  # Scale intensity based on red component

    ripe_orange_r = int(255 * scale)  # Full red for vibrance
    ripe_orange_g = int(ran * scale)  # Higher green for ripe appearance
    ripe_orange_b = int(20 * scale)  # Very minimal blue for vibrancy
    return (ripe_orange_r, ripe_orange_g, ripe_orange_b)


def adjust_to_bright_orange(r, g, b):
    """
    Adjusts the color to a brighter orange hue,
    with intensity proportional to the original red value.
    """
    scale = r / 255  # Scale intensity based on red component
    bright_orange_r = int(255 * scale)
    bright_orange_g = int(165 * scale)  # Increase green for a brighter orange
    bright_orange_b = int(50 * scale)  # Small blue component for brightness
    return (bright_orange_r, bright_orange_g, bright_orange_b)


def change_red_to_fruit_orange(image_path):
    """
    Changes all pixels that are shades of red to a fruit-orange hue,
    with intensity proportional to the original red value.
    """
    # Open the image
    img = Image.open(image_path)
    img = img.convert("RGB")  # Ensure the image is in RGB mode
    pixels = img.load()  # Access pixel data
    ran = random.randint(90, 150)
    print(ran)
    for x in range(img.width):
        for y in range(img.height):
            r, g, b = pixels[x, y]
            if is_shade_of_red(r, g, b):
                # Adjust the color to a fruit-orange hue
                pixels[x, y] = adjust_to_ripe_orange(r, g, b, ran)
                # pixels[x, y] = adjust_to_bright_orange(r, g, b)

    # Save the output image
    return img
    i  # mg.save(output_path)
    # print(f"Modified image saved as {output_path}")


# Example usage
def process_images_in_folder(input_folder, output_folder):
    """
    Processes all images in the specified input folder, applies the orange filter,
    and saves the modified images into the specified output folder.
    """
   # if not os.path.exists(output_folder):
    #    os.makedirs(output_folder)  # Create the output folder if it doesn't exist

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Filter image files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Apply the orange filter
            filtered_image = change_red_to_fruit_orange(input_path)

            # Save the modified image to the output folder
            filtered_image.save(output_path)
            print(f"Processed and saved: {output_path}")


# Specify the input folder containing the images and the output folder ("trainA")
# Replace with the path to your input folder
# input_folder = "path/to/your/images"
# output_folder = os.path.join(input_folder, "trainA")

# Process the images
process_images_in_folder("../apple2orange/trainA", "trainA")

# change_red_to_fruit_orange(
#   "../apple2orange/trainA/n07740461_15196.jpg", "output_image.jpg")
