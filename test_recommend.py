import json
import torch
import numpy as np
import re
from tqdm import tqdm
from datasets import load_dataset
import os
from collections import defaultdict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    LogitsProcessorList, 
    LogitsProcessor
)

class APIRecommendationTester:
    def __init__(self, model_path, api_repo_json=None, api_data_json=None, device=None):
        """
        Initialize tester for API recommendation models with reason validation
        
        Args:
            model_path: Path to the trained model
            api_repo_json: Path to the API repository JSON file (optional)
            api_data_json: Path to the API data JSONL file for descriptions
            device: Torch device to use (if None, will use CUDA if available)
        """
        self.model_path = model_path
        self.api_repo_json = api_repo_json
        self.api_data_json = api_data_json
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model components
        self.model = None
        self.tokenizer = None
        
        # Special tokens
        self.API_START_TOKEN = "<API_START>"
        self.API_STOP_TOKEN = "<API_STOP>"
        self.REASON_START_TOKEN = "<REASON_START>"
        self.REASON_STOP_TOKEN = "<REASON_STOP>"
        
        # API mappings
        self.api_special_tokens = {}
        self.special_token_to_api = {}
        self.api_list = []
        self.api_descriptions = {}  # Store API descriptions
        
        # Evaluation tracking
        self.empty_predictions = 0
        self.duplicate_predictions = 0
        self.missing_reasons = 0
        self.incomplete_reasons = 0
        
        # Load model and initialize
        self._load_model()
        if api_repo_json:
            self._load_api_repository()
        if api_data_json:
            self._load_api_descriptions()
        
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            device_map={"": self.device}
        ).to(self.device)
        
        # Get special token IDs
        self.special_token_ids = {
            "API_START_ID": self.tokenizer.convert_tokens_to_ids(self.API_START_TOKEN),
            "API_STOP_ID": self.tokenizer.convert_tokens_to_ids(self.API_STOP_TOKEN),
            "REASON_START_ID": self.tokenizer.convert_tokens_to_ids(self.REASON_START_TOKEN),
            "REASON_STOP_ID": self.tokenizer.convert_tokens_to_ids(self.REASON_STOP_TOKEN),
            "EOS_TOKEN_ID": self.tokenizer.eos_token_id
        }
        
        print("Model and tokenizer loaded successfully")
        
    def _load_api_repository(self):
        """Load API repository from JSON file"""
        if not os.path.exists(self.api_repo_json):
            print(f"API repository file not found: {self.api_repo_json}")
            return
            
        with open(self.api_repo_json, 'r', encoding='utf-8') as f:
            api_data = json.load(f)
            
      
        if isinstance(api_data, list):
            self.api_list = [item['name'] if isinstance(item, dict) else item for item in api_data]
        else:
            self.api_list = list(api_data.keys())
            
  
        for idx, name in enumerate(self.api_list):
            token = f"<API_{name}>"
            self.api_special_tokens[name] = token
            self.special_token_to_api[token] = name
            
        print(f"Loaded {len(self.api_list)} APIs from repository")
        
        
        
    def generate_recommendation_with_reason(self, mashup_description, max_length=1000):

        # Create the prompt exactly as in training
        prompt = (
                "API Recommendation Task"
                "Recommend APIs for the mashup according to its description and give the reason for recommendation"
                f"Mashup Description: {mashup_description}"
                f"Recommended APIs: {self.API_START_TOKEN}"

            )
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                do_sample=False,
            
                repetition_penalty=1.5,
                # top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        generated_part = generated_text[len(input_text):].strip()
        
        print(generated_part)
        
        # Parse the generated output
        result = {
            "target_apis": [],
            "reason": "",
            "raw_output": generated_part,
            "has_reason": False,
            "reason_complete": False,
            "apis_with_reasons": {}
        }
        
        try:
            # Extract Target APIs - look for content between API_START and API_STOP tokens
            if self.API_STOP_TOKEN in generated_part:
                api_section = generated_part.split(self.API_STOP_TOKEN)[0].strip()
                # Extract API tokens
                api_tokens = re.findall(r'<API_[^>]+>', api_section)
                result["target_apis"] = [
                    self.special_token_to_api.get(token, token.replace('<API_', '').replace('>', '')) 
                    for token in api_tokens
                ]
            
            # Extract Reason - look for content between REASON_START and REASON_STOP tokens
            if self.REASON_START_TOKEN in generated_part:
                result["has_reason"] = True
                reason_part = generated_part.split(self.REASON_START_TOKEN)[1]
                if self.REASON_STOP_TOKEN in reason_part:
                    reason_section = reason_part.split(self.REASON_STOP_TOKEN)[0].strip()
                    result["reason_complete"] = True
                else:
                    reason_section = reason_part.strip()
                    result["reason_complete"] = False
                    
                result["reason"] = reason_section
                
                # Check if each recommended API has a corresponding reason mentioned
                result["apis_with_reasons"] = self._analyze_api_reasons(
                    result["target_apis"], result["reason"]
                )
                
        except Exception as e:
            print(f"Error parsing generated output: {e}")
            print(f"Generated part: {generated_part}")
        
        return result
    def _analyze_api_reasons(self, recommended_apis, reason_text):
        """Analyze which APIs are mentioned in the reason text"""
        apis_with_reasons = {}
        
        for api in recommended_apis:
            # Check if API is mentioned in reason (case-insensitive)
            api_mentioned = False
            
            # Check direct mention
            if api.lower() in reason_text.lower():
                api_mentioned = True
            
            # Check API token mention
            api_token = f"<API_{api}>"
            if api_token in reason_text:
                api_mentioned = True
            

            apis_with_reasons[api] = api_mentioned
            
        return apis_with_reasons
        
    def validate_predictions(self, api_names):
        """Validate API predictions by checking for empty results or duplicates"""
        # Check if prediction list is empty
        has_empty = len(api_names) == 0
        if has_empty:
            self.empty_predictions += 1
            
        # Check for duplicates
        unique_apis = []
        seen = set()
        for api in api_names:
            if api not in seen:
                seen.add(api)
                unique_apis.append(api)
                
        has_duplicates = len(unique_apis) < len(api_names)
        if has_duplicates:
            self.duplicate_predictions += 1
            
        return has_empty, has_duplicates, unique_apis
        
    def validate_reasons(self, result):
        """Validate reason completeness"""
        if not result["has_reason"]:
            self.missing_reasons += 1
            return False, "No reason provided"
            
        if not result["reason_complete"]:
            self.incomplete_reasons += 1
            return False, "Reason section incomplete"
            
        # Check if all APIs have reasons mentioned
        apis_without_reasons = [
            api for api, has_reason in result["apis_with_reasons"].items() 
            if not has_reason
        ]
        
        if apis_without_reasons:
            return False, f"APIs without reasons: {apis_without_reasons}"
            
        return True, "All APIs have reasons"
        
    def format_reason_with_api_markers(self, reason, predicted_apis):
        """Format reason text to mark APIs with <API_...> markers (keep original format)"""
        formatted_reason = reason
        
        for api in predicted_apis:
            # no variations 
            api_variations = [
                api
            ]
            
            for variation in api_variations:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variation) + r'\b'
                replacement = f"<API_{api}>"
                # Only replace if it's not already in API format
                if f"<API_{api}>" not in formatted_reason:
                    formatted_reason = re.sub(pattern, replacement, formatted_reason, flags=re.IGNORECASE)

        return formatted_reason
        
    def create_reason_evaluation_entry(self, prompt, predicted_apis, reason):
        """Create an entry for evaluate_data_reason.jsonl"""
        entry = {
            "prompt": prompt,
            "completion": predicted_apis,
            "target_api_description": "",
            "reason": ""
        }
        
        # Build target_api_description from predicted APIs
        api_descriptions = []
        for api in predicted_apis:
            description = self.api_descriptions.get(api, f"No description available for {api}")
            api_descriptions.append(f"{api}: {description}")
        
        entry["target_api_description"] = " ".join(api_descriptions)
        
        # Format reason with API markers (keep <API_...> format)
        entry["reason"] = self.format_reason_with_api_markers(reason, predicted_apis)
        
        return entry
        
    # Metric calculation functions (from evaluator)
    def ndcg(self, target, pred):
        """Calculate NDCG@k"""
        if len(pred) == 0:
            return 0.0
        
        dcg = 0
        c = 0
        for i in range(1, len(pred) + 1):
            rel = 0
            if pred[i - 1] in target:
                rel = 1
                c += 1
            dcg += (np.power(2, rel) - 1) / np.log2(i + 1)
        
        if c == 0:
            return 0.0
        
        # 关键修改：c 不应该超过真实标签数量，避免 IDCG 计算偏差
        # 这个注意是错的c = min(len(target), c)
        c = min(len(target), len(pred))
        idcg = 0
        for i in range(1, c + 1):
            idcg += (1 / np.log2(i + 1))
        
        return dcg / idcg

    def ap(self, target, pred):
        """Calculate average precision"""
        if len(pred) == 0:
            return 0.0
        p_at_k = np.zeros(len(pred))
        c = 0
        for i in range(1, len(pred) + 1):
            rel = 0
            if pred[i - 1] in target:
                rel = 1
                c += 1
            p_at_k[i - 1] = rel * c / i
        if c == 0:
            return 0.0
        else:
            # 这个是错的return np.sum(p_at_k) / c
            return np.sum(p_at_k) / len(target)
    def precision(self, target, pred):
        """Calculate precision"""
        if len(pred) == 0:
            return 0.0
        hit_set = list(set(target) & set(pred))
        return len(hit_set) / float(len(pred))

    def recall(self, target, pred):
        """Calculate recall"""
        if len(target) == 0:
            return 0.0
        hit_set = list(set(target) & set(pred))
        return len(hit_set) / float(len(target))

    def calculate_metrics(self, ground_truth, predictions, k=None):
        """Calculate all metrics for a single prediction"""
        if k is not None:
            preds = predictions[:k]
        else:
            preds = predictions
            
        if len(preds) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg": 0.0,
                "map": 0.0
            }
            
        precision = self.precision(ground_truth, preds)
        recall = self.recall(ground_truth, preds)
        ndcg_val = self.ndcg(ground_truth, preds)
        ap_val = self.ap(ground_truth, preds)
        
        return {
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg_val,
            "map": ap_val
        }
        
    def test_model(self, test_data_path, output_path=None, reason_output_path="evaluate_data_reason.jsonl", top_k_list=[1, 3, 5, 10]):
        """
        Test model on test data with comprehensive evaluation including reason validation
        
        Args:
            test_data_path: Path to test data file
            output_path: Path for saving results (optional)
            reason_output_path: Path for saving reason evaluation data
            top_k_list: List of k values for top-k metrics
            
        Returns:
            Dictionary of comprehensive test results
        """
        print("\n=== Testing API Recommendation Model with Reason Validation ===")
        
        # Load test data
        if os.path.exists(test_data_path):
            data = load_dataset("json", data_files=test_data_path)["train"]
            print(f"Loaded {len(data)} test samples from {test_data_path}")
        else:
            raise FileNotFoundError(f"Test data file not found: {test_data_path}")
            
        # Create API mappings from data if no repository provided
        if not self.api_list:
            self.create_api_mappings_from_data(test_data_path)
        
        # Reset counters
        self.empty_predictions = 0
        self.duplicate_predictions = 0 
        self.missing_reasons = 0
        self.incomplete_reasons = 0
        
        # Results structure
        results = {
            "samples": [],
            "metrics": {},
            "top_k": {k: {"precision": [], "recall": [], "ndcg": [], "map": []} for k in top_k_list},
            "dynamic_k": {"precision": [], "recall": [], "ndcg": [], "map": [], "k_values": []},
            "reason_validation": {
                "missing_reasons": 0,
                "incomplete_reasons": 0,
                "apis_without_reasons": 0,
                "total_apis_recommended": 0,
                "apis_with_reasons": 0,
                "reason_quality_examples": []
            },
            "prediction_validation": {
                "empty_count": 0,
                "duplicate_count": 0
            }
        }
        
        # Open reason evaluation file for writing
        reason_file = open(reason_output_path, 'w', encoding='utf-8')
        
        # Process test data
        for item in tqdm(data, desc="Testing model"):
            # Extract mashup description and ground truth
            mashup = item.get("prompt", "")
            ground_truth = item.get("completion", [])
            
            # Generate recommendation with reason
            result = self.generate_recommendation_with_reason(mashup, max_length=800)
            api_predictions = result["target_apis"]
            # actually recommended_apis
            # Validate predictions
            is_empty, has_duplicates, unique_apis = self.validate_predictions(api_predictions)
            
            # Validate reasons
            reason_valid, reason_message = self.validate_reasons(result)
            
            # Create and write reason evaluation entry if we have predictions and reason
            if unique_apis and result["reason"]:
                reason_entry = self.create_reason_evaluation_entry(
                    mashup, unique_apis, result["reason"]
                )
                reason_file.write(json.dumps(reason_entry, ensure_ascii=False) + '\n')
                reason_file.flush()  # Ensure immediate write
            
            # Update reason validation stats
            results["reason_validation"]["total_apis_recommended"] += len(unique_apis)
            
            for api, has_reason in result["apis_with_reasons"].items():
                if has_reason:
                    results["reason_validation"]["apis_with_reasons"] += 1
                else:
                    results["reason_validation"]["apis_without_reasons"] += 1
            
            if not result["has_reason"]:
                results["reason_validation"]["missing_reasons"] += 1
            elif not result["reason_complete"]:
                results["reason_validation"]["incomplete_reasons"] += 1
                
            # Update prediction validation stats
            if is_empty:
                results["prediction_validation"]["empty_count"] += 1
            if has_duplicates:
                results["prediction_validation"]["duplicate_count"] += 1
            
            # Record dynamic k values
            results["dynamic_k"]["k_values"].append(len(unique_apis))
            
            # Calculate metrics for fixed k values
            for k in top_k_list:
                metrics_at_k = self.calculate_metrics(ground_truth, unique_apis, k)
                for metric_name, value in metrics_at_k.items():
                    results["top_k"][k][metric_name].append(value)
            
            # Calculate metrics for dynamic k
            dynamic_metrics = self.calculate_metrics(ground_truth, unique_apis)
            for metric_name, value in dynamic_metrics.items():
                results["dynamic_k"][metric_name].append(value)
            
            # Store sample results
            sample_result = {
                "input": mashup,
                "predictions": unique_apis,
                "ground_truth": ground_truth,
                "reason": result["reason"],
                "has_reason": result["has_reason"],
                "reason_complete": result["reason_complete"],
                "apis_with_reasons": result["apis_with_reasons"],
                "reason_valid": reason_valid,
                "reason_message": reason_message,
                "empty": is_empty,
                "has_duplicates": has_duplicates,
                "raw_output": result["raw_output"]
            }
            results["samples"].append(sample_result)
            
            # Collect examples for reason quality analysis
            if len(results["reason_validation"]["reason_quality_examples"]) < 10:
                results["reason_validation"]["reason_quality_examples"].append({
                    "mashup": mashup,
                    "apis": unique_apis,
                    "reason": result["reason"],
                    "apis_with_reasons": result["apis_with_reasons"],
                    "reason_valid": reason_valid
                })
        
        # Close reason file
        reason_file.close()
        print(f"Reason evaluation data saved to {reason_output_path}")
        
        # Calculate average metrics
        total_samples = len(data)
        
        # Fixed k metrics
        for k in top_k_list:
            for metric_name in ["precision", "recall", "ndcg", "map"]:
                values = results["top_k"][k][metric_name]
                results["metrics"][f"{metric_name}@{k}"] = np.mean(values)
        
        # Dynamic k metrics
        avg_k = np.mean(results["dynamic_k"]["k_values"])
        results["metrics"]["avg_k"] = avg_k
        
        for metric_name in ["precision", "recall", "ndcg", "map"]:
            values = results["dynamic_k"][metric_name]
            results["metrics"][f"{metric_name}@dynamic"] = np.mean(values)
        
        # Validation metrics
        results["metrics"]["empty_prediction_rate"] = results["prediction_validation"]["empty_count"] / total_samples
        results["metrics"]["duplicate_prediction_rate"] = results["prediction_validation"]["duplicate_count"] / total_samples
        results["metrics"]["missing_reason_rate"] = results["reason_validation"]["missing_reasons"] / total_samples
        results["metrics"]["incomplete_reason_rate"] = results["reason_validation"]["incomplete_reasons"] / total_samples
        
        # Reason coverage metrics
        total_apis = results["reason_validation"]["total_apis_recommended"]
        if total_apis > 0:
            results["metrics"]["api_reason_coverage"] = results["reason_validation"]["apis_with_reasons"] / total_apis
        else:
            results["metrics"]["api_reason_coverage"] = 0.0
        
        # Print comprehensive results
        self._print_test_results(results)
        
        # Save results if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Detailed results saved to {output_path}")
            
        return results
        
    def _print_test_results(self, results):
        """Print comprehensive test results"""
        print("\n=== Comprehensive Test Results ===")
        
        # Basic metrics
        print(f"Average number of APIs recommended: {results['metrics']['avg_k']:.2f}")
        
        # Prediction validation
        print(f"\n--- Prediction Validation ---")
        print(f"Empty predictions: {results['prediction_validation']['empty_count']} ({results['metrics']['empty_prediction_rate']:.2%})")
        print(f"Predictions with duplicates: {results['prediction_validation']['duplicate_count']} ({results['metrics']['duplicate_prediction_rate']:.2%})")
        
        # Reason validation 
        print(f"\n--- Reason Validation ---")
        print(f"Samples without reasons: {results['reason_validation']['missing_reasons']} ({results['metrics']['missing_reason_rate']:.2%})")
        print(f"Samples with incomplete reasons: {results['reason_validation']['incomplete_reasons']} ({results['metrics']['incomplete_reason_rate']:.2%})")
        print(f"API-reason coverage: {results['metrics']['api_reason_coverage']:.2%}")
        print(f"Total APIs recommended: {results['reason_validation']['total_apis_recommended']}")
        print(f"APIs with reasons mentioned: {results['reason_validation']['apis_with_reasons']}")
        print(f"APIs without reasons: {results['reason_validation']['apis_without_reasons']}")
        
        # Performance metrics
        print(f"\n--- Performance Metrics ---")
        for k in [1, 3, 5, 10]:
            if f"precision@{k}" in results["metrics"]:
                print(f"Top-{k} metrics:")
                print(f"  Precision@{k}: {results['metrics'][f'precision@{k}']:.4f}")
                print(f"  Recall@{k}: {results['metrics'][f'recall@{k}']:.4f}")
                print(f"  NDCG@{k}: {results['metrics'][f'ndcg@{k}']:.4f}")
                print(f"  MAP@{k}: {results['metrics'][f'map@{k}']:.4f}")
                
        print("Dynamic k metrics:")
        print(f"  Precision@dynamic: {results['metrics']['precision@dynamic']:.4f}")
        print(f"  Recall@dynamic: {results['metrics']['recall@dynamic']:.4f}")
        print(f"  NDCG@dynamic: {results['metrics']['ndcg@dynamic']:.4f}")
        print(f"  MAP@dynamic: {results['metrics']['map@dynamic']:.4f}")
        
        # Show some examples
        print(f"\n--- Reason Quality Examples ---")
        for i, example in enumerate(results["reason_validation"]["reason_quality_examples"][:3]):
            print(f"\nExample {i+1}:")
            print(f"Mashup: {example['mashup'][:100]}...")
            print(f"APIs: {example['apis']}")
            print(f"Reason: {example['reason'][:200]}...")
            print(f"APIs with reasons: {example['apis_with_reasons']}")
            print(f"Valid: {example['reason_valid']}")


# Example usage
if __name__ == "__main__":
    # Initialize tester
    tester = APIRecommendationTester(
        model_path="api_grpo_test/train",  
        api_repo_json="used_api_list.json",  
        api_data_json="api_data.jsonl",  
        device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    )
    
    results = tester.test_model(
        test_data_path="test_data.jsonl",  
        output_path="test_results/comprehensive_test_results.json",
        reason_output_path="evaluate_data_reason_train.jsonl", 
        top_k_list=[1, 3, 5, 10]
    )
    
    print("\n=== Test Complete ===")
    print(f"Overall API-reason coverage: {results['metrics']['api_reason_coverage']:.2%}")
    print(f"Average recommendation performance (Precision@5): {results['metrics']['precision@5']:.4f}")
    print("Reason evaluation data generated in evaluate_data_reason.jsonl")
