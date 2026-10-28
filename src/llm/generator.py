import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time

class LocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", load_in_4bit: bool = False):
        self.model_name = model_name
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "generation_seconds": 0.0}
        
        # Determine device and dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model loading kwargs
        kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
        }
        
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        
    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.1) -> str:
        # Chat templates are essential for instruct models like Qwen
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        start = time.perf_counter()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += int(model_inputs.input_ids.shape[-1])
        self.stats["completion_tokens"] += len(generated_ids[0])
        self.stats["generation_seconds"] += time.perf_counter() - start
        return response.strip()

    def stats_snapshot(self):
        return dict(self.stats)

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global instance
_llm_instance = None

def get_llm(model_name: str = "Qwen/Qwen2.5-3B-Instruct", load_in_4bit: bool = False) -> LocalLLM:
    global _llm_instance
    if _llm_instance is None or _llm_instance.model_name != model_name:
        if _llm_instance is not None:
            _llm_instance.cleanup()
        _llm_instance = LocalLLM(model_name, load_in_4bit)
    return _llm_instance
