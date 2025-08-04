import json
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ReasonQualityAnalyzer:
    def __init__(self):
        """Initialize the analyzer for RP and RR statistics"""
        self.stats = {
            'rp_scores': [],
            'rr_scores': [],
            'reason_rewards': [],
            'api_counts': [],
            'mentioned_api_counts': [],
            'intersection_counts': [],
            'no_recommendation_count': 0,
            'no_reason_count': 0,
            'perfect_rp_count': 0,
            'perfect_rr_count': 0,
            'perfect_both_count': 0,
            'zero_rp_count': 0,
            'zero_rr_count': 0,
            'samples': []
        }
    
    def extract_apis_from_reason(self, reason_text):
        """Extract API names from reason text using <API_...> pattern"""
        if not reason_text:
            return set()
        
        # Extract APIs wrapped in <API_...> tags
        api_pattern = r'<API_([^>]+)>'
        matches = re.findall(api_pattern, reason_text)
        
        # Clean and return unique APIs
        apis = set()
        for match in matches:
            api_name = match.strip()
            if api_name:
                apis.add(api_name)
        
        return apis
    
    def compute_reason_quality_metrics(self, recommended_apis, generated_reason):
        """Compute RP, RR, and reason reward as per the paper formulation"""
        # Extract APIs mentioned in reason (APIs_res)
        reason_mentioned_apis = self.extract_apis_from_reason(generated_reason) if generated_reason else set()
        
        # Convert to sets for computation
        APIs_rec = set(recommended_apis) if recommended_apis else set()  # Recommended APIs
        APIs_res = reason_mentioned_apis  # APIs mentioned in reason
        
        # Compute intersection
        intersection = APIs_rec.intersection(APIs_res)
        
        # Compute Reason Precision (RP) = |APIs_rec ∩ APIs_res| / |APIs_res|
        if len(APIs_res) > 0:
            RP = len(intersection) / len(APIs_res)
        else:
            RP = 0.0  # No APIs mentioned in reason
        
        # Compute Reason Recall (RR) = |APIs_rec ∩ APIs_res| / |APIs_rec|
        if len(APIs_rec) > 0:
            RR = len(intersection) / len(APIs_rec)
        else:
            RR = 0.0  # No recommended APIs
        
        # Combined reason reward: reason_reward = 0.5 × RP + 0.5 × RR
        reason_reward = 0.5 * RP + 0.5 * RR
        
        return RP, RR, reason_reward, len(APIs_rec), len(APIs_res), len(intersection)
    
    def analyze_dataset(self, jsonl_file_path):
        """Analyze the entire GRPO dataset"""
        print(f"Analyzing dataset: {jsonl_file_path}")
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total samples: {len(lines)}")
        
        for line_idx, line in enumerate(tqdm(lines, desc="Processing samples")):
            try:
                data = json.loads(line.strip())
                
                # Extract recommended APIs (from completion field)
                recommended_apis = data.get('completion', [])
                if isinstance(recommended_apis, str):
                    recommended_apis = [recommended_apis]
                
                # Extract reason text
                reason_text = data.get('reason', '')
                
                # Compute metrics
                RP, RR, reason_reward, APIs_rec_size, APIs_res_size, intersection_size = self.compute_reason_quality_metrics(
                    recommended_apis, reason_text
                )
                
                # Store statistics
                self.stats['rp_scores'].append(RP)
                self.stats['rr_scores'].append(RR)
                self.stats['reason_rewards'].append(reason_reward)
                self.stats['api_counts'].append(APIs_rec_size)
                self.stats['mentioned_api_counts'].append(APIs_res_size)
                self.stats['intersection_counts'].append(intersection_size)
                
                # Count special cases
                if APIs_rec_size == 0:
                    self.stats['no_recommendation_count'] += 1
                if APIs_res_size == 0:
                    self.stats['no_reason_count'] += 1
                if RP == 1.0:
                    self.stats['perfect_rp_count'] += 1
                if RR == 1.0:
                    self.stats['perfect_rr_count'] += 1
                if RP == 1.0 and RR == 1.0:
                    self.stats['perfect_both_count'] += 1
                if RP == 0.0:
                    self.stats['zero_rp_count'] += 1
                if RR == 0.0:
                    self.stats['zero_rr_count'] += 1
                
                # Store sample for detailed analysis
                sample_info = {
                    'line_idx': line_idx,
                    'prompt': data.get('prompt', '')[:100] + '...' if len(data.get('prompt', '')) > 100 else data.get('prompt', ''),
                    'recommended_apis': recommended_apis,
                    'reason_apis': list(self.extract_apis_from_reason(reason_text)),
                    'RP': RP,
                    'RR': RR,
                    'reason_reward': reason_reward,
                    'APIs_rec_size': APIs_rec_size,
                    'APIs_res_size': APIs_res_size,
                    'intersection_size': intersection_size
                }
                self.stats['samples'].append(sample_info)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_idx}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_idx}: {e}")
                continue
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        total_samples = len(self.stats['rp_scores'])
        
        print("\n" + "="*60)
        print("REASON QUALITY STATISTICS ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Samples with no recommendations: {self.stats['no_recommendation_count']} ({self.stats['no_recommendation_count']/total_samples:.2%})")
        print(f"Samples with no reason APIs: {self.stats['no_reason_count']} ({self.stats['no_reason_count']/total_samples:.2%})")
        
        # RP Statistics
        print(f"\nReason Precision (RP) Statistics:")
        print(f"Mean RP: {np.mean(self.stats['rp_scores']):.4f}")
        print(f"Median RP: {np.median(self.stats['rp_scores']):.4f}")
        print(f"Std RP: {np.std(self.stats['rp_scores']):.4f}")
        print(f"Min RP: {np.min(self.stats['rp_scores']):.4f}")
        print(f"Max RP: {np.max(self.stats['rp_scores']):.4f}")
        print(f"Perfect RP (1.0): {self.stats['perfect_rp_count']} ({self.stats['perfect_rp_count']/total_samples:.2%})")
        print(f"Zero RP (0.0): {self.stats['zero_rp_count']} ({self.stats['zero_rp_count']/total_samples:.2%})")
        
        # RR Statistics
        print(f"\nReason Recall (RR) Statistics:")
        print(f"Mean RR: {np.mean(self.stats['rr_scores']):.4f}")
        print(f"Median RR: {np.median(self.stats['rr_scores']):.4f}")
        print(f"Std RR: {np.std(self.stats['rr_scores']):.4f}")
        print(f"Min RR: {np.min(self.stats['rr_scores']):.4f}")
        print(f"Max RR: {np.max(self.stats['rr_scores']):.4f}")
        print(f"Perfect RR (1.0): {self.stats['perfect_rr_count']} ({self.stats['perfect_rr_count']/total_samples:.2%})")
        print(f"Zero RR (0.0): {self.stats['zero_rr_count']} ({self.stats['zero_rr_count']/total_samples:.2%})")
        
        # Combined Perfect Statistics
        print(f"\nCombined Perfect Statistics:")
        print(f"Perfect Both RP & RR (1.0): {self.stats['perfect_both_count']} ({self.stats['perfect_both_count']/total_samples:.2%})")
        
        # Reason Reward Statistics
        print(f"\nReason Reward Statistics:")
        print(f"Mean Reason Reward: {np.mean(self.stats['reason_rewards']):.4f}")
        print(f"Median Reason Reward: {np.median(self.stats['reason_rewards']):.4f}")
        print(f"Std Reason Reward: {np.std(self.stats['reason_rewards']):.4f}")
        print(f"Min Reason Reward: {np.min(self.stats['reason_rewards']):.4f}")
        print(f"Max Reason Reward: {np.max(self.stats['reason_rewards']):.4f}")
        
        # API Count Statistics
        print(f"\nAPI Count Statistics:")
        print(f"Mean recommended APIs per sample: {np.mean(self.stats['api_counts']):.2f}")
        print(f"Mean mentioned APIs per sample: {np.mean(self.stats['mentioned_api_counts']):.2f}")
        print(f"Mean intersection size: {np.mean(self.stats['intersection_counts']):.2f}")
        
        # Distribution analysis
        print(f"\nDistribution Analysis:")
        rp_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        rp_hist, _ = np.histogram(self.stats['rp_scores'], bins=rp_bins)
        rr_hist, _ = np.histogram(self.stats['rr_scores'], bins=rp_bins)
        
        print("RP Distribution:")
        for i in range(len(rp_bins)-1):
            print(f"  {rp_bins[i]:.1f}-{rp_bins[i+1]:.1f}: {rp_hist[i]} ({rp_hist[i]/total_samples:.2%})")
        
        print("RR Distribution:")
        for i in range(len(rp_bins)-1):
            print(f"  {rp_bins[i]:.1f}-{rp_bins[i+1]:.1f}: {rr_hist[i]} ({rr_hist[i]/total_samples:.2%})")
    
    def show_examples(self, num_examples=5):
        """Show examples of different RP/RR scenarios"""
        print(f"\n" + "="*60)
        print("EXAMPLE ANALYSIS")
        print("="*60)
        
        # Sort samples by different criteria
        samples = self.stats['samples']
        
        # High RP, High RR examples
        high_rp_rr = [s for s in samples if s['RP'] > 0.8 and s['RR'] > 0.8]
        if high_rp_rr:
            print(f"\nHigh RP & RR Examples (RP > 0.8, RR > 0.8): {len(high_rp_rr)} samples")
            for i, sample in enumerate(high_rp_rr[:num_examples]):
                print(f"  Example {i+1}:")
                print(f"    Prompt: {sample['prompt']}")
                print(f"    Recommended APIs: {sample['recommended_apis']}")
                print(f"    Reason APIs: {sample['reason_apis']}")
                print(f"    RP: {sample['RP']:.3f}, RR: {sample['RR']:.3f}, Reward: {sample['reason_reward']:.3f}")
        
        # Low RP, Low RR examples
        low_rp_rr = [s for s in samples if s['RP'] < 0.2 and s['RR'] < 0.2]
        if low_rp_rr:
            print(f"\nLow RP & RR Examples (RP < 0.2, RR < 0.2): {len(low_rp_rr)} samples")
            for i, sample in enumerate(low_rp_rr[:num_examples]):
                print(f"  Example {i+1}:")
                print(f"    Prompt: {sample['prompt']}")
                print(f"    Recommended APIs: {sample['recommended_apis']}")
                print(f"    Reason APIs: {sample['reason_apis']}")
                print(f"    RP: {sample['RP']:.3f}, RR: {sample['RR']:.3f}, Reward: {sample['reason_reward']:.3f}")
        

        high_rp_low_rr = [s for s in samples if s['RP'] > 0.8 and s['RR'] < 0.2]
        if high_rp_low_rr:
            print(f"\nHigh RP, Low RR Examples (RP > 0.8, RR < 0.2): {len(high_rp_low_rr)} samples")
            for i, sample in enumerate(high_rp_low_rr[:num_examples]):
                print(f"  Example {i+1}:")
                print(f"    Prompt: {sample['prompt']}")
                print(f"    Recommended APIs: {sample['recommended_apis']}")
                print(f"    Reason APIs: {sample['reason_apis']}")
                print(f"    RP: {sample['RP']:.3f}, RR: {sample['RR']:.3f}, Reward: {sample['reason_reward']:.3f}")
        

        low_rp_high_rr = [s for s in samples if s['RP'] < 0.2 and s['RR'] > 0.8]
        if low_rp_high_rr:
            print(f"\nLow RP, High RR Examples (RP < 0.2, RR > 0.8): {len(low_rp_high_rr)} samples")
            for i, sample in enumerate(low_rp_high_rr[:num_examples]):
                print(f"  Example {i+1}:")
                print(f"    Prompt: {sample['prompt']}")
                print(f"    Recommended APIs: {sample['recommended_apis']}")
                print(f"    Reason APIs: {sample['reason_apis']}")
                print(f"    RP: {sample['RP']:.3f}, RR: {sample['RR']:.3f}, Reward: {sample['reason_reward']:.3f}")
    
    def save_detailed_analysis(self, output_file):
        """Save detailed analysis to JSON file"""
        analysis_data = {
            'statistics': {
                'total_samples': len(self.stats['rp_scores']),  # Fixed: was 'js_scores'
                'no_recommendation_count': self.stats['no_recommendation_count'],
                'no_reason_count': self.stats['no_reason_count'],
                'rp_stats': {
                    'mean': float(np.mean(self.stats['rp_scores'])),
                    'median': float(np.median(self.stats['rp_scores'])),
                    'std': float(np.std(self.stats['rp_scores'])),
                    'min': float(np.min(self.stats['rp_scores'])),
                    'max': float(np.max(self.stats['rp_scores'])),
                    'perfect_count': self.stats['perfect_rp_count'],
                    'zero_count': self.stats['zero_rp_count']
                },
                'rr_stats': {
                    'mean': float(np.mean(self.stats['rr_scores'])),
                    'median': float(np.median(self.stats['rr_scores'])),
                    'std': float(np.std(self.stats['rr_scores'])),
                    'min': float(np.min(self.stats['rr_scores'])),
                    'max': float(np.max(self.stats['rr_scores'])),
                    'perfect_count': self.stats['perfect_rr_count'],
                    'zero_count': self.stats['zero_rr_count']
                },
                'perfect_both_count': self.stats['perfect_both_count'],
                'reason_reward_stats': {
                    'mean': float(np.mean(self.stats['reason_rewards'])),
                    'median': float(np.median(self.stats['reason_rewards'])),
                    'std': float(np.std(self.stats['reason_rewards'])),
                    'min': float(np.min(self.stats['reason_rewards'])),
                    'max': float(np.max(self.stats['reason_rewards']))
                }
            },
            'samples': self.stats['samples']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":

    analyzer = ReasonQualityAnalyzer()  
    
    analyzer.analyze_dataset("ds.jsonl")

    analyzer.print_statistics()
    
    analyzer.show_examples(num_examples=3)
    analyzer.save_detailed_analysis("reason_quality_analysis.json")  
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)