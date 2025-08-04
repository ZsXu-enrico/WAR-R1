import os
import json
import random
import re
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
    """
    Similar to HuggingFace's DataCollator, but does not perform padding,
    directly wraps a single sample as batch_size=1.
    """
    def __call__(self, features):
        # Only supports batch_size=1
        if len(features) != 1:
            raise ValueError(f"NoOpDataCollator only supports batch_size=1, got {len(features)}")
        feature = features[0]
        out = {}
        for k, v in feature.items():
            if k == "target_apis":
                # If there are special fields that don't need tensorization, keep them as is
                out[k] = v
            else:
                # Convert list or 1D tensor to LongTensor, then unsqueeze(0)
                if isinstance(v, list):
                    t = torch.tensor(v, dtype=torch.long)
                elif isinstance(v, torch.Tensor):
                    t = v
                else:
                    raise ValueError(f"Unexpected feature type for key {k}: {type(v)}")
                out[k] = t.unsqueeze(0)  
        
        return out

class APIStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for API generation"""
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is the stop token
        if input_ids[0, -1] == self.stop_token_id:
            return True
        return False
class APIRecommendationTrainer:
    def __init__(
        self,
        base_model_name: str = "tiny-llama",
        device: torch.device = None
    ):
        # Device setup
        self.DEVICE = torch.device("cuda:3")
        print(f"Using device: {self.DEVICE}")

        # Tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, padding=False, truncation=True, max_length=1000
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map={"": self.DEVICE},
            attn_implementation="eager",  # Explicitly use eager implementation instead of flash attention
        ).to(self.DEVICE)

        # Special tokens
        self.EOS_TOKEN = self.tokenizer.eos_token
        self.STOP_NAME = "STOP"
        self.API_STOP_TOKEN = f"<API_{self.STOP_NAME}>"
        
        # Add API_START token
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

        # Maps
        self.api_special_tokens = {}
        self.special_token_to_api = {}
        self.api_to_index = {}
        self.index_to_api = {}
        self.api_token_ids = []
        self.num_apis = 0

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
            
        # Ensure api_token_ids is on the correct device
        self.api_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(tok) for tok in tokens],
            device=self.DEVICE
        )
        self.num_apis = len(tokens)
        print(f"Added {self.num_apis} api tokens, vocab size={len(self.tokenizer)}")
        return names

    def extract_apis_from_data(self, raw_ds: Dataset):
        """Extract unique API names from the training data (from completion and reason fields)"""
        api_names = set()
        
        for example in raw_ds:
            # Extract from completion field
            if "completion" in example and isinstance(example["completion"], list):
                for api in example["completion"]:
                    api_names.add(api)
            elif "completion" in example and isinstance(example["completion"], str):
                api_names.add(example["completion"])
            
            # Extract from reason field (APIs wrapped with ***)
            if "reason" in example and isinstance(example["reason"], str):
                reason_apis = re.findall(r'\*\*\*([^*]+)\*\*\*', example["reason"])
                for api in reason_apis:
                    api_names.add(api.strip())
        
        return list(api_names)

    def create_api_tokens_from_data(self, raw_ds: Dataset):
        """Create API special tokens from the training data"""
        api_names = self.extract_apis_from_data(raw_ds)
        
        tokens = [f"<API_{name}>" for name in api_names]
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

        for idx, name in enumerate(api_names):
            tok = tokens[idx]
            self.api_special_tokens[name] = tok
            self.special_token_to_api[tok] = name
            self.api_to_index[name] = idx
            self.index_to_api[idx] = name
            
        # Ensure api_token_ids is on the correct device
        self.api_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(tok) for tok in tokens],
            device=self.DEVICE
        )
        self.num_apis = len(tokens)
        print(f"Created {self.num_apis} api tokens from training data, vocab size={len(self.tokenizer)}")
        return api_names

    def convert_reason_apis_to_tokens(self, reason_text):
        """Convert APIs wrapped with *** in reason text to API special tokens"""
        if not isinstance(reason_text, str):
            return reason_text
        
        def replace_api(match):
            api_name = match.group(1).strip()
            return self.api_special_tokens.get(api_name, f"<API_{api_name}>")
        
        # Replace ***api_name*** with <API_api_name>
        converted_reason = re.sub(r'\*\*\*([^*]+)\*\*\*', replace_api, reason_text)
        return converted_reason

    def prepare_dataset_phase1(self, raw_ds: Dataset, max_length: int = 512):
        """Prepare dataset for autoregressive training - compute loss on the entire sequence"""
        processed_data = []
        
        for example in raw_ds:
            mashup_description = example["prompt"]
            target_apis = example["completion"]
            reason = example.get("reason", "")
            
            # Convert API names to special tokens
            if isinstance(target_apis, list):
                api_tokens = [self.api_special_tokens.get(api, f"<API_{api}>") for api in target_apis]
                api_tokens_str = " ".join(api_tokens)
            else:
                api_tokens_str = self.api_special_tokens.get(target_apis, f"<API_{target_apis}>")
            
            # Convert reason APIs (***api***) to special tokens
            converted_reason = self.convert_reason_apis_to_tokens(reason)
            
            # Complete training text (input + expected output)
            full_text = (
                "API Recommendation Task"
                "Recommend APIs for the mashup according to its description and give the reason for each recommendation"
                f"Mashup Description: {mashup_description}"
                f"Recommended APIs: {self.API_START_TOKEN}{api_tokens_str}{self.API_STOP_TOKEN}"
                f"Reason: {self.REASON_START_TOKEN}{converted_reason}{self.REASON_STOP_TOKEN}{self.EOS_TOKEN}"
            )
            
            # Tokenize the complete text
            encoding = self.tokenizer(
                full_text, 
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            # print(input_ids)
            # For causal LM, labels are the same as input_ids
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

    def compute_phase1_loss(self, outputs, inputs):
        """Phase 1: Autoregressive training with LM loss"""
        return outputs.loss

    # Define custom trainer for Phase 1
    class Phase1Trainer(Trainer):
        """Custom trainer for Phase 1: Autoregressive training with LM loss"""
        def __init__(self, api_trainer, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.api_trainer = api_trainer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Move tensors to DEVICE before passing to model
            inputs_on_device = {k: v.to(self.api_trainer.DEVICE) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
            
    
            model.to(self.api_trainer.DEVICE)
            

            outputs = model(**{k: v for k, v in inputs_on_device.items() if k in ["input_ids", "attention_mask", "labels"]})
            
            # In Phase 1, we use LM loss
            loss = self.api_trainer.compute_phase1_loss(outputs, inputs_on_device)
            

            loss = loss.to(torch.device("cuda:0"))
            return (loss, outputs) if return_outputs else loss

    def generate_recommendation(self, model, mashup_description, max_length=1000):
        """Generate API recommendation and reason for a given mashup description"""
        # Create the prompt exactly as in training (but without the expected output)
        prompt = (
                "API Recommendation Task"
                "Recommend APIs for the mashup according to its description and give the reason for recommendation"
                f"Mashup Description: {mashup_description}"
                f"Recommended APIs: {self.API_START_TOKEN}"
            )
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs["input_ids"].to(self.DEVICE)
        attention_mask = inputs["attention_mask"].to(self.DEVICE)
        
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                do_sample=True,
                temperature=1,
                # top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the generated part (remove the original prompt)
        if prompt in generated_text:
            generated_part = generated_text[len(prompt):].strip()
        else:
            generated_part = generated_text
        
        # Parse the generated output
        result = {
            "target_apis": [],
            "reason": "",
            "raw_output": generated_part
        }
        
        try:
            # Extract Target APIs - look for content between API_START and API_STOP tokens
            if self.API_STOP_TOKEN in generated_part:
                api_section = generated_part.split(self.API_STOP_TOKEN)[0].strip()
                # Extract API tokens
                api_tokens = re.findall(r'<API_[^>]+>', api_section)
                result["target_apis"] = [self.special_token_to_api.get(token, token.replace('<API_', '').replace('>', '')) for token in api_tokens]
            
            # Extract Reason - look for content between REASON_START and REASON_STOP tokens
            if self.REASON_START_TOKEN in generated_part:
                reason_part = generated_part.split(self.REASON_START_TOKEN)[1]
                if self.REASON_STOP_TOKEN in reason_part:
                    reason_section = reason_part.split(self.REASON_STOP_TOKEN)[0].strip()
                else:
                    reason_section = reason_part.strip()
                result["reason"] = reason_section
                
        except Exception as e:
            print(f"Error parsing generated output: {e}")
            print(f"Generated part: {generated_part}")
        
        return result

    def test_generation(self, model, test_descriptions):
        """Test the model with multiple mashup descriptions"""
        print("\n=== Testing API Recommendation Generation ===")
        for desc in test_descriptions:
            print(f"\nMashup Description: {desc}")
            result = self.generate_recommendation(model, desc)
            print(f"Recommended APIs: {result['target_apis']}")
            print(f"Reason: {result['reason']}")
            if result.get('raw_output'):
                print(f"Raw Output: {result['raw_output']}")
            print("-" * 80)
        
    def train(
        self,
        api_repo: str,
        train_path: str,
        output_dir: str,
        phase1_epochs: int = 10,
        lr: float = 1e-5
    ):
        os.makedirs(output_dir, exist_ok=True)
        

        raw = load_dataset("json", data_files=train_path)["train"]
        
        # Create API tokens from training data if no API repo provided
        if api_repo and os.path.exists(api_repo):
            self.load_and_add_api_tokens(api_repo)
        else:
            print("No API repo provided, creating tokens from training data...")
            self.create_api_tokens_from_data(raw)
        
        # Phase 1: Autoregressive training
        print("=== Starting Phase 1: Autoregressive Training (Full Task Learning) ===")
        phase1_ds = self.prepare_dataset_phase1(raw)
        
        phase1_args = TrainingArguments(
            output_dir=f"{output_dir}/phase1",
            num_train_epochs=phase1_epochs,
            per_device_train_batch_size=1,  
            learning_rate=lr,
            save_steps=5000,
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
        
        # Test after phase 1 with multiple examples
        test_mashups = [
            "let tour internet via skype bot",
            "create a weather app with location services",
            "build a social media dashboard with analytics",
            "develop a travel booking system"
        ]
        
        self.test_generation(self.model, test_mashups)

if __name__ == "__main__":
    APIER = APIRecommendationTrainer()
    APIER.train(
        "used_api_list.json",
        "train_data_reason_ds.jsonl", 
        "api_reason_ds", 
        phase1_epochs=20
    )
