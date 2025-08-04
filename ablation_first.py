import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList
)
from datasets import load_dataset, Dataset

class NoOpDataCollator:
 
    def __call__(self, features):
        if len(features) != 1:
            raise ValueError(f"NoOpDataCollator only supports batch_size=1, got {len(features)}")
        feature = features[0]
        out = {}
        for k, v in feature.items():
            if k == "target_apis":
                out[k] = v
            else:
                if isinstance(v, list):
                    t = torch.tensor(v, dtype=torch.long)
                elif isinstance(v, torch.Tensor):
                    t = v
                else:
                    raise ValueError(f"Unexpected feature type for key {k}: {type(v)}")
                out[k] = t.unsqueeze(0)
        return out

class APIStoppingCriteria(StoppingCriteria):
   
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0, -1] == self.stop_token_id:
            return True
        return False

class APIRecommendationTrainer:
    def __init__(self, base_model_name: str = "tiny-llama", device: str = "cuda:0"):
       
        self.DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.DEVICE}")

        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, padding=False, truncation=True, max_length=1000
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map={"": self.DEVICE},
            attn_implementation="eager",
        ).to(self.DEVICE)

        self.EOS_TOKEN = self.tokenizer.eos_token
        self.API_STOP_TOKEN = "<API_STOP>"
        self.DES_START_TOKEN = "<DES_START>"
        self.DES_STOP_TOKEN = "<DES_STOP>"
        self.API_START_TOKEN = "<API_START>"
        
        
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                self.API_START_TOKEN, self.API_STOP_TOKEN,
                self.DES_START_TOKEN, self.DES_STOP_TOKEN,
            ]
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

    
        self.API_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.API_STOP_TOKEN)
        self.DES_START_ID = self.tokenizer.convert_tokens_to_ids(self.DES_START_TOKEN)
        self.DES_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.DES_STOP_TOKEN)
        
        print(f"Added special tokens, vocab size: {len(self.tokenizer)}")
        
        self.api_special_tokens = {}
        self.special_token_to_api = {}
        self.api_token_ids = []
        self.num_apis = 0

   
        self.prompt_template = (
            "### API Function Description Task\n"
            "API_name: {api}\n"
            "Give the function description of the API: " + self.DES_START_TOKEN + "{completion}" + self.DES_STOP_TOKEN
            + self.EOS_TOKEN
        )

    def load_and_add_api_tokens(self, api_repo_path: str):
  
        with open(api_repo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = [item["name"] if isinstance(item, dict) else item for item in data]

        tokens = [f"<API_{n}>" for n in names]
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

        for idx, name in enumerate(names):
            tok = tokens[idx]
            self.api_special_tokens[name] = tok
            self.special_token_to_api[tok] = name
            
        self.api_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(tok) for tok in tokens],
            device=self.DEVICE
        )
        self.num_apis = len(tokens)
        print(f"Added {self.num_apis} API tokens, vocab size: {len(self.tokenizer)}")
        return names

    def format_example_phase1(self, example):
        
        api_name = example["prompt"]
        api_des = example["completion"]
        api_token = f"<API_{api_name}>"
        text = self.prompt_template.format(api=api_token, completion=api_des)
        return text

    def prepare_dataset_phase1(self, raw_ds: Dataset, max_length: int = 400):
    
        items = [self.format_example_phase1(i) for i in raw_ds]
        processed_data = []
        
        for text in items: 
            encodings = self.tokenizer(text, truncation=True, padding=False, max_length=max_length)
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            labels = input_ids.copy()
            
            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        
        return Dataset.from_dict({
            "input_ids": [item["input_ids"] for item in processed_data],
            "attention_mask": [item["attention_mask"] for item in processed_data],
            "labels": [item["labels"] for item in processed_data],
        })

    class Phase1Trainer(Trainer):
        def __init__(self, api_trainer, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.api_trainer = api_trainer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs_on_device = {k: v.to(self.api_trainer.DEVICE) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
            
            model.to(self.api_trainer.DEVICE)
            outputs = model(**{k: v for k, v in inputs_on_device.items() if k in ["input_ids", "attention_mask", "labels"]})
            loss = outputs.loss
            
            return (loss, outputs) if return_outputs else loss

    def generate_description(self, model, api_name, max_length=1000):

        if not api_name.startswith("<API_"):
            api_token = f"<API_{api_name}>"
        else:
            api_token = api_name
  
        prompt = (
            "### API Function Description Task\n"
            f"API_name: {api_token}\n"
            "Give the function description of the API: " + self.DES_START_TOKEN
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs["input_ids"].to(self.DEVICE)
        attention_mask = inputs["attention_mask"].to(self.DEVICE)
        
        stopping_criteria = StoppingCriteriaList([APIStoppingCriteria(self.DES_STOP_ID)])
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.DES_STOP_ID,
                stopping_criteria=stopping_criteria
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if self.DES_START_TOKEN in generated_text:
            description_part = generated_text.split(self.DES_START_TOKEN)[1]
            if self.DES_STOP_TOKEN in description_part:
                description = description_part.split(self.DES_STOP_TOKEN)[0].strip()
            else:
                description = description_part.strip()
        else:
            description = "No description generated"
        
        return description
        
    def train(self, api_repo: str, train_path: str, output_dir: str, phase1_epochs: int = 10, lr: float = 1e-5):

        os.makedirs(output_dir, exist_ok=True)
        
        if api_repo and os.path.exists(api_repo):
            self.load_and_add_api_tokens(api_repo)
        
        raw = load_dataset("json", data_files=train_path)["train"]
        
        print("=== Starting Phase 1: Autoregressive Training ===")
        phase1_ds = self.prepare_dataset_phase1(raw)
        
        phase1_args = TrainingArguments(
            output_dir=f"{output_dir}/phase1",
            num_train_epochs=phase1_epochs,
            per_device_train_batch_size=1,
            learning_rate=lr,
            save_steps=1000,
            logging_steps=1000
        )
        
        phase1_trainer = self.Phase1Trainer(
            self,
            model=self.model,
            args=phase1_args,
            train_dataset=phase1_ds,
            tokenizer=self.tokenizer,
            data_collator=NoOpDataCollator()
        )
        
        phase1_trainer.train()
        phase1_trainer.save_model(phase1_args.output_dir)
        
       
        test_api_name = "pubnub-javascript-push"
        print("=== Phase 1 Test ===")
        result = self.generate_description(self.model, test_api_name)
        print(f"Generated description for {test_api_name}: {result}")

if __name__ == "__main__":
    trainer = APIRecommendationTrainer(device="cuda:0")  
    trainer.train(
        "used_api_list.json",  
        "api_data.jsonl", 
        "api_knowledge", 
        phase1_epochs=20
    )