# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from transformers import FuyuForCausalLM, AutoTokenizer, FuyuProcessor, FuyuImageProcessor

MODEL_NAME = "adept/fuyu-8b"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=TOKEN_CACHE,
        )
        image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(
            image_processor=image_processor,
            tokenizer=self.tokenizer
        )
        self.model = FuyuForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            device_map="cuda"
        )

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        image: Path = Input(
            description="Input Image", 
        ),
        max_new_tokens: int = Input(
            description="Max new tokens",
            default=512,
            le=2048,
            ge=0,
        ),
    ) -> str:
        """Run a single prediction on the model"""
        image_pil = Image.open(image)
        full_prompt = prompt+"\n"

        model_inputs = self.processor(
            text=full_prompt,
            images=[image_pil],
            device="cuda"
        )
        for k, v in model_inputs.items():
            model_inputs[k] = v.to("cuda")
        
        generation_output = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        neg_tok = -1*max_new_tokens
        generation_text = self.processor.batch_decode(
            generation_output[:, neg_tok:],
            skip_special_tokens=True
        )
        gen_text = generation_text[0]
        index = gen_text.find(prompt)
        result = gen_text[index + len(prompt):].strip()
        return result

