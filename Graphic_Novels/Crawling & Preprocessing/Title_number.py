from PIL import Image
import re
import os

# Function to crop the image into the specified number of cuts
def crop_image(image, cuts,box_size=(70, 70)):
    width_old, height_old = image.size

    # Crop the image except the top 40 pixels
    cut_img = image.crop((0, 40, width_old, height_old))

    width, height = cut_img.size

    # Calculate the size of the horizontal and vertical divisions into two.
    half_width = width // 2
    half_height = height // 6

    # Prepare a list containing the images to be segmented.
    images_splitted = []

    # Split the image into four square regions.
    for i in range(2):
        for j in range(int(cuts/2)):
            # Calculate the starting coordinates of each rectangular area.
            left = i * half_width
            upper = j * half_height
            # Calculate the end coordinates of each rectangular area.
            right = (i + 1) * half_width
            lower = (j + 1) * half_height
            # Crop the image to a defined area.
            cropped_image = cut_img.crop((left, upper, right, lower))

            # Create a white box
            white_box = Image.new("RGB", box_size, "white")
            # Attach a box to the top right of the image
            cropped_image.paste(white_box, (half_width - box_size[0], 0))

            # Add the cropped image to the list.
            images_splitted.append(cropped_image)
    
    return images_splitted

# Get the paths to all image files in the folder.
directory_path = 'your_path'
s_path = 'your_path'
# Edit this section to filter only specific extensions.
file_extension = ['.jpeg', '.gif', '.jpg', '.png']

# Gets all file names within a specified folder.
#file_names = [f for f in os.listdir(directory_path)]# if f.endswith(file_extension)]

# Get a list of files within a directory, filter them by extension, and sort them by file name.
# Function to extract numbers from file names
def extract_numbers(s):
    return list(map(int, re.findall(r'\d+', s)))
# Get a list of files in a directory, filter them by extension, extract numbers using regular expressions, and sort them.
file_names = sorted(
    [f for f in os.listdir(directory_path) if any(f.lower().endswith(ext) for ext in file_extension)],
    key=extract_numbers
)

# Crop and save images
for index, file_name in enumerate(os.listdir(directory_path)):
    if any(file_name.lower().endswith(ext) for ext in file_extension):
        file_path = os.path.join(directory_path, file_name)
        image = Image.open(file_path)
        cropped_images = crop_image(image, 12)

        # make folders including index
        new_folder_name = f"{file_name.split('.')[0]}"
        new_folder_path = os.path.join(s_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)  # make folder if there is no 

        # save cropped images
        for i, img in enumerate(cropped_images):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            save_path = os.path.join(new_folder_path, f'cropped_{i+1}.jpg')
            img.save(save_path)

