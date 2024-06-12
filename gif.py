import imageio.v2 as imageio
import os

# Path to the directory containing PNG files
directory = 'temp/filling_injection/profound_20_DV3'

# List all PNG files in the directory and sort them alphabetically
filenames = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png')])
print(filenames)

# Read images
images = []
for i, filename in enumerate(filenames):
    if 'None' not in filename and i < len(filenames) - 1 and 'None' in filenames[i + 1]:
        print(filename)
        for _ in range(5):
            images.append(imageio.imread(filename))
    else:
        images.append(imageio.imread(filename))

# Create a GIF
output_path = os.path.join(directory, 'filling_injection.gif')
imageio.mimsave(output_path, images, format='gif', fps=1.5, loop=0)

print(f"GIF created at {output_path}")
