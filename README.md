# Adept/Fuyu-8b Cog model

This is an implementation of the [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="Generate a coco-style caption." -i image=@bus.png

## Example:

Input:

"Generate a coco-style caption."

Output:

"A bus parked on the side of a road."
