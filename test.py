from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor
from PIL import Image
import torch

MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

# load model, tokenizer, and processor
pretrained_path = "adept/fuyu-8b"
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_path,
    cache_dir=TOKEN_CACHE
)

image_processor = FuyuImageProcessor()
processor = FuyuProcessor(
    image_processor=image_processor, tokenizer=tokenizer)

model = FuyuForCausalLM.from_pretrained(
    pretrained_path,
    device_map="cuda",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
)

# test inference
text_prompt = "Generate a coco-style caption."
image_path = "bus.png"  # https://huggingface.co/adept-hf-collab/fuyu-8b/blob/main/bus.png
image_pil = Image.open(image_path)
text_prompt += "\n"

model_inputs = processor(
    text=text_prompt,
    images=[image_pil],
    device="cuda"
)
for k, v in model_inputs.items():
    model_inputs[k] = v.to("cuda")

generation_output = model.generate(**model_inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
print(generation_text[0])