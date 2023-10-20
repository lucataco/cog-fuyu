# Adept/Fuyu-8b Cog model

This is an implementation of the [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="What is the highest life expectancy at birth of male?" -i image=@chart.png

## Example:

Input

"What is the highest life expectancy at birth of male?"

![alt text](chart.png)

Output

The life expectancy at birth of males in 2018 is 80.7.
