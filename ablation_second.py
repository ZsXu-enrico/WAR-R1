import os
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
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

class APIRecommendationTrainer:
    def __init__(
        self,
        base_model_name: str = "api_knowledge/phase1",
        device: torch.device = None
    ):
        self.DEVICE = torch.device("cuda") if device is None else device
        print(f"Using device: {self.DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, padding=False, truncation=True, max_length=400
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
        self.STOP_NAME = "STOP"
        self.API_STOP_TOKEN = f"<API_{self.STOP_NAME}>"
        
        self.DES_START_TOKEN = "<DES_START>"
        self.DES_STOP_TOKEN = "<DES_STOP>"
        self.API_START_TOKEN = "<API_START>"
        self.REASON_START_TOKEN = "<REASON_START>"
        self.REASON_STOP_TOKEN = "<REASON_STOP>"
        
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                self.API_START_TOKEN, self.API_STOP_TOKEN,
                self.DES_START_TOKEN, self.DES_STOP_TOKEN,
                self.REASON_START_TOKEN, self.REASON_STOP_TOKEN,
            ]
        })
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.API_START_ID = self.tokenizer.convert_tokens_to_ids(self.API_START_TOKEN)
        self.API_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.API_STOP_TOKEN)
        self.DES_START_ID = self.tokenizer.convert_tokens_to_ids(self.DES_START_TOKEN)
        self.DES_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.DES_STOP_TOKEN)
        self.REASON_START_ID = self.tokenizer.convert_tokens_to_ids(self.REASON_START_TOKEN)
        self.REASON_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.REASON_STOP_TOKEN)
        
        print(f"Added API_START token: {self.API_START_TOKEN}, ID: {self.API_START_ID}")
        print(f"Added API_STOP token: {self.API_STOP_TOKEN}, ID: {self.API_STOP_ID}")
        print(f"Added DES_START token: {self.DES_START_TOKEN}, ID: {self.DES_START_ID}")
        print(f"Added DES_STOP token: {self.DES_STOP_TOKEN}, ID: {self.DES_STOP_ID}")
        print(f"Added REASON_START token: {self.REASON_START_TOKEN}, ID: {self.REASON_START_ID}")
        print(f"Added REASON_STOP token: {self.REASON_STOP_TOKEN}, ID: {self.REASON_STOP_ID}")

        self.api_special_tokens = {}
        self.special_token_to_api = {}
        self.api_to_index = {}
        self.index_to_api = {}
        self.api_token_ids = []
        self.num_apis = 0

        self.prompt_template = (
            "### API Recommendation Task\n"
            "Recommend APIs for the mashup according to its description\n"
            "Mashup: {mashup}\n"
            "Recommended APIs: " + self.API_START_TOKEN + "{completion}" + self.API_STOP_TOKEN
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
            self.api_to_index[name] = idx
            self.index_to_api[idx] = name
            
        self.api_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(tok) for tok in tokens],
            device=self.DEVICE
        )
        self.num_apis = len(tokens)
        print(f"Added {self.num_apis} api tokens, vocab size={len(self.tokenizer)}")
        return names

    def format_example_phase1(self, example):
        mashup = json.dumps(example["prompt"], ensure_ascii=False)
        apis = example.get("completion", [])
        
        tokens = [self.api_special_tokens.get(a) for a in apis if self.api_special_tokens.get(a) is not None]
        text = self.prompt_template.format(mashup=mashup, completion="".join(tokens))
        return text, apis

    def prepare_dataset_phase1(self, raw_ds: Dataset, max_length: int = 400):
        items = [self.format_example_phase1(i) for i in raw_ds]
        texts, targets = zip(*items) if items else ([], [])

        processed_data = []
        
        for text, target_apis in zip(texts, targets):
            encodings = self.tokenizer(text, truncation=False, padding=False)
            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"]
            
            api_start_pos = -1
            for idx, token_id in enumerate(input_ids):
                if token_id == self.API_START_ID:
                    api_start_pos = idx
                    break
            
            labels = input_ids.copy()
            
            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "target_apis": target_apis
            })
        
        if not processed_data:
            return Dataset.from_dict({
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "target_apis": []
            })
            
        return Dataset.from_dict({
            "input_ids": [item["input_ids"] for item in processed_data],
            "attention_mask": [item["attention_mask"] for item in processed_data],
            "labels": [item["labels"] for item in processed_data],
            "target_apis": [item["target_apis"] for item in processed_data]
        })

    def compute_phase1_loss(self, outputs, inputs):
        return outputs.loss

    class Phase1Trainer(Trainer):
        def __init__(self, api_trainer, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.api_trainer = api_trainer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs_on_device = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs_on_device[k] = v.to(self.api_trainer.DEVICE)
                else:
                    inputs_on_device[k] = v
            
            model.to(self.api_trainer.DEVICE)
            
            forward_inputs = {k: v for k, v in inputs_on_device.items() 
                            if k in ["input_ids", "attention_mask", "labels"]}
            outputs = model(**forward_inputs)
            
            loss = self.api_trainer.compute_phase1_loss(outputs, inputs_on_device)
            
            if hasattr(loss, 'to'):  
                loss = loss.to(torch.device("cuda:0"))
            
            return (loss, outputs) if return_outputs else loss

    def predict_apis_generate(self, model, text, max_new_tokens=100):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        gen_ids = outputs[0][inputs.input_ids.size(1):].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(gen_ids)
        apis = [self.special_token_to_api[t] for t in tokens if t in self.special_token_to_api]
        
        if self.STOP_NAME in apis:
            apis.remove(self.STOP_NAME)
            
        return apis

    def generate_recommendations(self, model, mashup_description):
        prompt = (
            "### API Recommendation Task\n"
            "Recommend APIs for the mashup according to its description\n"
            f"Mashup Description: {mashup_description}\n"
            f"Recommended APIs: {self.API_START_TOKEN}"
        )

        print("=== Generate() Recommendations ===")
        gen_apis = self.predict_apis_generate(model, prompt)
        print(gen_apis)

        return {"generate": gen_apis}
        
    def train(
        self,
        api_repo: str,
        train_path: str,
        output_dir: str,
        phase1_epochs: int = 10,
        lr: float = 1e-5
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.load_and_add_api_tokens(api_repo)
        raw = load_dataset("json", data_files=train_path)["train"]
        
        print("=== Starting Phase 1: Autoregressive Training with LM Loss ===")
        phase1_ds = self.prepare_dataset_phase1(raw)
        
        phase1_args = TrainingArguments(
            output_dir=f"{output_dir}/phase1",
            num_train_epochs=phase1_epochs,
            per_device_train_batch_size=1,
            learning_rate=lr,
            save_steps=24655,
            logging_steps=1000,
            dataloader_pin_memory=False,
            remove_unused_columns=False
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
        
        test_description = "using service search local event evdb cell phone wap sm email toolbar using 411sync mobile api"
        
        model_1_path = f"{output_dir}/phase1"
        try:
            model_1 = AutoModelForCausalLM.from_pretrained(
                model_1_path,
                torch_dtype=torch.float32,
                device_map={"": self.DEVICE}
            ).to(self.DEVICE)
            
            print("=== Phase 1 Test Recommendations ===")
            phase1_results = self.generate_recommendations(model_1, test_description)
            print(phase1_results)
        except Exception as e:
            print(f"Error loading trained model: {e}")

if __name__ == "__main__":
    APIER = APIRecommendationTrainer()
    APIER.train(
        "used_api_list.json", 
        "train_data_reason_ds.jsonl", 
        "api_recommender", 
        phase1_epochs=20
    )