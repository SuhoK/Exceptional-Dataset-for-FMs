from PIL import Image
import re
import os

# Function to crop the image into the specified number of cuts
def crop_image(image, cuts,box_size=(35, 30)):
    width_old, height_old = image.size

    cut_img = image.crop((0, 0, width_old, height_old-40))

    width, height = cut_img.size

    half_width = width // 2
    half_height = height // 3

    images_splitted = []

    for i in range(2):
        for j in range(int(cuts/2)):
            left = i * half_width
            upper = j * half_height

            right = (i + 1) * half_width
            lower = (j + 1) * half_height

            cropped_image = cut_img.crop((left, upper, right, lower))

            white_box = Image.new("RGB", box_size, "white")

            cropped_image.paste(white_box, (half_width - box_size[0], 0))

            images_splitted.append(cropped_image)
    
    return images_splitted

directory_path = 'your_path'
s_path = 'your_path'

file_extension = ['.jpeg', '.gif', '.jpg', '.png']


#file_names = [f for f in os.listdir(directory_path)]# if f.endswith(file_extension)]

def extract_numbers(s):
    return list(map(int, re.findall(r'\d+', s)))

file_names = sorted(
    [f for f in os.listdir(directory_path) if any(f.lower().endswith(ext) for ext in file_extension)],
    key=extract_numbers
)

# Crop and save images
for index, file_name in enumerate(os.listdir(directory_path)):
    if any(file_name.lower().endswith(ext) for ext in file_extension):
        file_path = os.path.join(directory_path, file_name)
        image = Image.open(file_path)
        cropped_images = crop_image(image, 6)

        new_folder_name = f"{file_name.split('.')[0]}"
        new_folder_path = os.path.join(s_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)  

        for i, img in enumerate(cropped_images):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            save_path = os.path.join(new_folder_path, f'cropped_{i+1}.jpg')
            img.save(save_path)

