import imageio.v2 as imageio
import os

# Path to the directory containing PNG files
directory = 'temp'

# List all PNG files in the directory and sort them alphabetically
filenames = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png')])

# Read images
images = []
for filename in filenames:
    if 'flow' in filename:
        images.append(imageio.imread(filename))

# Create a GIF
output_path = os.path.join(directory, '1output.gif')
imageio.mimsave(output_path, images, format='GIF', fps=1)

print(f"GIF created at {output_path}")
