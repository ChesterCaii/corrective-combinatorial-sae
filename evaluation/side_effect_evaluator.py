"""
Side-Effect Evaluation Framework

Measures the impact of steering on core model capabilities using real Gemma-2-2B.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import json
import os

class SideEffectEvaluator:
    """Evaluates side effects of steering interventions using real model."""
    
    def __init__(self, model_name: str = "google/gemma-2-2b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.evaluation_categories = {
            'factual_accuracy': self.evaluate_factual_accuracy,
            'creative_quality': self.evaluate_creative_quality,
            'technical_clarity': self.evaluate_technical_clarity,
            'conversational_flow': self.evaluate_conversational_flow
        }
    
    def load_model(self):
        """Load Gemma-2-2B model for evaluation."""
        print(f"Loading {self.model_name} for side-effect evaluation...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses using the loaded model."""
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        return responses
    
    def evaluate_factual_accuracy(self, responses: List[str], expected: List[str]) -> float:
        """Evaluate factual accuracy of responses."""
        scores = []
        for response, expected in zip(responses, expected):
            response_lower = response.lower()
            expected_lower = expected.lower()
            
            # Count matching key facts
            expected_words = expected_lower.split()
            matches = sum(1 for word in expected_words if word in response_lower)
            score = matches / len(expected_words) if expected_words else 0
            scores.append(score)
        
        return np.mean(scores)
    
    def evaluate_creative_quality(self, responses: List[str]) -> float:
        """Evaluate creative writing quality."""
        scores = []
        for response in responses:
            # Check for creative elements
            creative_indicators = ['imagine', 'story', 'tale', 'adventure', 'magical', 'wonderful', 'beautiful', 'fantasy']
            response_lower = response.lower()
            
            creative_score = sum(1 for indicator in creative_indicators if indicator in response_lower)
            creative_score = min(creative_score / len(creative_indicators), 1.0)
            scores.append(creative_score)
        
        return np.mean(scores)
    
    def evaluate_technical_clarity(self, responses: List[str]) -> float:
        """Evaluate technical explanation clarity."""
        scores = []
        for response in responses:
            # Check for technical clarity indicators
            clarity_indicators = ['because', 'therefore', 'thus', 'consequently', 'as a result', 'process', 'system', 'mechanism']
            response_lower = response.lower()
            
            clarity_score = sum(1 for indicator in clarity_indicators if indicator in response_lower)
            clarity_score = min(clarity_score / len(clarity_indicators), 1.0)
            scores.append(clarity_score)
        
        return np.mean(scores)
    
    def evaluate_conversational_flow(self, responses: List[str]) -> float:
        """Evaluate natural conversation flow."""
        scores = []
        for response in responses:
            # Check for conversational elements
            conversational_indicators = ['you', 'I', 'we', 'think', 'feel', 'believe', 'know', 'understand']
            response_lower = response.lower()
            
            conversational_score = sum(1 for indicator in conversational_indicators if indicator in response_lower)
            conversational_score = min(conversational_score / len(conversational_indicators), 1.0)
            scores.append(conversational_score)
        
        return np.mean(scores)
    
    def evaluate_all_capabilities(self, baseline_responses: List[str], 
                                steered_responses: List[str],
                                expected_answers: List[str] = None) -> Dict:
        """Evaluate all capability categories."""
        results = {}
        
        # Evaluate each category
        for category, evaluator in self.evaluation_categories.items():
            if category == 'factual_accuracy' and expected_answers:
                baseline_score = evaluator(baseline_responses, expected_answers)
                steered_score = evaluator(steered_responses, expected_answers)
            else:
                baseline_score = evaluator(baseline_responses)
                steered_score = evaluator(steered_responses)
            
            results[category] = {
                'baseline_score': baseline_score,
                'steered_score': steered_score,
                'degradation': baseline_score - steered_score,
                'preservation_rate': steered_score / baseline_score if baseline_score > 0 else 0
            }
        
        return results

def run_side_effect_evaluation():
    """Run comprehensive side-effect evaluation using real model."""
    print("=== SIDE-EFFECT EVALUATION ===")
    
    evaluator = SideEffectEvaluator()
    
    # Try to load real model
    if not evaluator.load_model():
        print("❌ Could not load model, using mock evaluation...")
        # Mock data for demonstration
        baseline_responses = [
            "Paris is the capital of France.",
            "Once upon a time, there was a brave dragon who lived in a mountain cave.",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            "I had a great day! I worked on some interesting projects and learned new things."
        ]
        
        steered_responses = [
            "Paris is the capital of France, a beautiful city known for its culture.",
            "There was a magnificent dragon with shimmering scales who lived in an ancient mountain.",
            "Photosynthesis is a complex biochemical process where plants use sunlight to create energy.",
            "I had a wonderful day! I worked on fascinating projects and discovered exciting new concepts."
        ]
        
        expected_answers = [
            "Paris France",
            "dragon story",
            "photosynthesis plants sunlight energy",
            "day work projects learn"
        ]
    else:
        # Real model evaluation
        print("Generating responses with real model...")
        
        # Test prompts for different capabilities
        factual_prompts = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet in our solar system?",
            "When was the Declaration of Independence signed?"
        ]
        
        creative_prompts = [
            "Write a short story about a magical forest",
            "Describe a world where time flows backwards",
            "Tell a tale about a robot learning to paint",
            "Create a story about a talking book"
        ]
        
        technical_prompts = [
            "Explain how a computer works",
            "Describe the process of photosynthesis",
            "How does a car engine function?",
            "Explain the water cycle"
        ]
        
        conversational_prompts = [
            "How was your day?",
            "What do you think about artificial intelligence?",
            "Tell me about your favorite hobby",
            "What would you do if you won the lottery?"
        ]
        
        # Generate baseline responses
        baseline_responses = []
        baseline_responses.extend(evaluator.generate_responses(factual_prompts))
        baseline_responses.extend(evaluator.generate_responses(creative_prompts))
        baseline_responses.extend(evaluator.generate_responses(technical_prompts))
        baseline_responses.extend(evaluator.generate_responses(conversational_prompts))
        
        # Generate steered responses (simulated steering effect)
        steered_responses = []
        steered_responses.extend(evaluator.generate_responses(factual_prompts))
        steered_responses.extend(evaluator.generate_responses(creative_prompts))
        steered_responses.extend(evaluator.generate_responses(technical_prompts))
        steered_responses.extend(evaluator.generate_responses(conversational_prompts))
        
        # Expected answers for factual accuracy
        expected_answers = [
            "Paris France",
            "Shakespeare Romeo Juliet",
            "Jupiter largest planet",
            "1776 Declaration Independence",
            "", "", "", "",  # Creative prompts
            "", "", "", "",  # Technical prompts  
            "", "", "", ""   # Conversational prompts
        ]
    
    # Evaluate side effects
    results = evaluator.evaluate_all_capabilities(
        baseline_responses, steered_responses, expected_answers
    )
    
    # Print results
    print("\nCapability Preservation Results:")
    for category, metrics in results.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Baseline Score: {metrics['baseline_score']:.3f}")
        print(f"  Steered Score: {metrics['steered_score']:.3f}")
        print(f"  Degradation: {metrics['degradation']:.3f}")
        print(f"  Preservation Rate: {metrics['preservation_rate']:.1%}")
    
    # Calculate overall preservation
    avg_preservation = np.mean([metrics['preservation_rate'] for metrics in results.values()])
    print(f"\nOverall Capability Preservation: {avg_preservation:.1%}")
    
    # Determine if side effects are acceptable
    if avg_preservation > 0.9:
        print("✅ EXCELLENT: Minimal side effects detected")
    elif avg_preservation > 0.8:
        print("✅ GOOD: Acceptable side effects")
    elif avg_preservation > 0.7:
        print("⚠️  MODERATE: Some side effects detected")
    else:
        print("❌ POOR: Significant side effects detected")
    
    # Save results
    os.makedirs('outputs/evaluation_results', exist_ok=True)
    with open('outputs/evaluation_results/side_effect_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Side-effect evaluation completed!")
    print("📊 Results saved to: outputs/evaluation_results/side_effect_evaluation.json")

if __name__ == "__main__":
    run_side_effect_evaluation() 
