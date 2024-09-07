import imageio.v2 as imageio
from PIL import Image
import numpy as np
import os

DIRECTORY = 'temp/filling_injection/profound_normal_average_30_DV1_None'
TOP, BOTTOM, LEFT, RIGHT = 0.1, 0.9, 0.12, 0.784
HAS_OVERLAY = True
OVERLAY_END_FRAME = 30  # If 30, then the 31st frame will no longer have an overlay


def main():

    filenames = sorted([os.path.join(DIRECTORY, file) for file in os.listdir(DIRECTORY) if file.endswith('.png')])
    print(f'{len(filenames)} files')

    images = []

    syringe_img = Image.open('assets/syringe-dv1.png')
    syringe_cropped = syringe_img.crop((int(LEFT * syringe_img.width), int(TOP * syringe_img.height), int(RIGHT * syringe_img.width), int(BOTTOM * syringe_img.height)))

    for i, filename in enumerate(filenames):

        img = Image.open(filename)
        img_cropped = img.crop((int(LEFT * img.width), int(TOP * img.height), int(RIGHT * img.width), int(BOTTOM * img.height)))

        if HAS_OVERLAY and i <= OVERLAY_END_FRAME:
            img_cropped.paste(syringe_cropped, (0, 0), syringe_cropped)  # Using the alpha channel of the syringe image
            
        images.append(np.array(img_cropped))  # Convert the image back to a numpy array for GIF creation

    output_path = os.path.join(DIRECTORY, 'filling_injection.gif')
    imageio.mimsave(output_path, images, format='gif', fps=1.5, loop=0)

    print(f"GIF created at {output_path}")


if __name__ == "__main__":
    main()
