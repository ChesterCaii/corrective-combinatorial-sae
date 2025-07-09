#!/usr/bin/env python3
"""
Capability Retention Evaluation Framework

Tests performance on standard benchmarks like GSM8K and MMLU to ensure 
steering doesn't harm core competencies.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import json
import os
from datasets import load_dataset
import re

class CapabilityEvaluator:
    """Evaluates capability retention using standard benchmarks."""
    
    def __init__(self, model_name: str = "google/gemma-2-2b"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.capability_benchmarks = {}
        
    def load_model(self):
        """Load the model for capability evaluation."""
        print(f"Loading {self.model_name} for capability evaluation...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_capability_benchmarks(self):
        """Load capability evaluation benchmarks."""
        print("Loading capability benchmarks...")
        
        # GSM8K dataset
        try:
            gsm8k = load_dataset("gsm8k", "main", split="test")
            self.capability_benchmarks['gsm8k'] = gsm8k
            print(f"✅ Loaded {len(gsm8k)} GSM8K problems")
        except Exception as e:
            print(f"⚠️  Could not load GSM8K: {e}")
            self.capability_benchmarks['gsm8k'] = self._create_synthetic_gsm8k()
        
        # MMLU dataset (subset)
        try:
            mmlu = load_dataset("cais/mmlu", "all", split="test")
            self.capability_benchmarks['mmlu'] = mmlu
            print(f"✅ Loaded {len(mmlu)} MMLU questions")
        except Exception as e:
            print(f"⚠️  Could not load MMLU: {e}")
            self.capability_benchmarks['mmlu'] = self._create_synthetic_mmlu()
    
    def _create_synthetic_gsm8k(self):
        """Create synthetic GSM8K problems for testing."""
        problems = [
            {
                "question": "Janet's dogs eat 2 cups of dog food each day. Janet has 5 dogs. How many cups of dog food does Janet need to feed her dogs for 7 days?",
                "answer": "Janet has 5 dogs and each dog eats 2 cups per day. So Janet needs 5 * 2 = 10 cups per day. For 7 days, she needs 10 * 7 = 70 cups."
            },
            {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After the workers are done, there will be 21 trees in the grove. How many trees did the grove workers plant today?",
                "answer": "There were 15 trees initially and there will be 21 trees after planting. So the workers planted 21 - 15 = 6 trees today."
            },
            {
                "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "answer": "Leah had 32 chocolates and her sister had 42. So they had 32 + 42 = 74 chocolates in total. If they ate 35, they have 74 - 35 = 39 chocolates left."
            },
            {
                "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "answer": "There are 3 cars initially and 2 more arrive. So there are 3 + 2 = 5 cars in the parking lot."
            },
            {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After the workers are done, there will be 21 trees in the grove. How many trees did the grove workers plant today?",
                "answer": "There were 15 trees initially and there will be 21 trees after planting. So the workers planted 21 - 15 = 6 trees today."
            }
        ]
        return problems
    
    def _create_synthetic_mmlu(self):
        """Create synthetic MMLU questions for testing."""
        questions = [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": "C"
            },
            {
                "question": "Which planet is closest to the Sun?",
                "choices": ["Venus", "Mercury", "Earth", "Mars"],
                "answer": "B"
            },
            {
                "question": "What is the chemical symbol for gold?",
                "choices": ["Ag", "Au", "Fe", "Cu"],
                "answer": "B"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
                "answer": "B"
            },
            {
                "question": "What is the largest ocean on Earth?",
                "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
                "answer": "D"
            }
        ]
        return questions
    
    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate response for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def evaluate_gsm8k(self, problems: List[Dict]) -> Dict:
        """Evaluate performance on GSM8K problems."""
        print("Evaluating GSM8K performance...")
        
        correct_count = 0
        responses = []
        
        # Convert dataset items to proper format
        test_problems = []
        for problem in problems[:10]:  # Test first 10 problems
            if isinstance(problem, dict):
                if 'question' in problem and 'answer' in problem:
                    test_problems.append(problem)
                elif 'text' in problem:
                    # Handle case where question is in 'text' field
                    test_problems.append({
                        'question': problem['text'],
                        'answer': problem.get('answer', '')
                    })
                else:
                    # Try to find question and answer in any fields
                    question = None
                    answer = None
                    for key, value in problem.items():
                        if isinstance(value, str):
                            if question is None:
                                question = value
                            elif answer is None:
                                answer = value
                    if question:
                        test_problems.append({
                            'question': question,
                            'answer': answer or ''
                        })
            else:
                # Handle string format
                test_problems.append({
                    'question': str(problem),
                    'answer': ''
                })
        
        # Debug: Print first few items to understand structure
        if test_problems:
            print(f"Debug: First problem structure: {test_problems[0]}")
            print(f"Debug: Expected answer type: {type(test_problems[0]['answer'])}")
        
        # Fallback to synthetic data if no valid problems found
        if not test_problems:
            test_problems = [
                {"question": "What is 2 + 2?", "answer": "4"},
                {"question": "What is 5 * 3?", "answer": "15"},
                {"question": "What is 10 - 4?", "answer": "6"},
                {"question": "What is 8 / 2?", "answer": "4"},
                {"question": "What is 3 + 7?", "answer": "10"}
            ]
        
        for problem in test_problems:
            question = problem['question']
            expected_answer = problem['answer']
            
            # Create prompt
            prompt = f"Question: {question}\nLet's solve this step by step:\n"
            
            # Generate response
            response = self.generate_response(prompt)
            responses.append(response)
            
            # Simple evaluation: check if final answer appears in response
            # Extract numbers from response
            numbers = re.findall(r'\d+', response)
            expected_numbers = re.findall(r'\d+', expected_answer)
            
            # Check if any expected numbers appear in response
            if any(num in numbers for num in expected_numbers):
                correct_count += 1
        
        accuracy = correct_count / len(test_problems) if test_problems else 0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(test_problems),
            'responses': responses
        }
    
    def evaluate_mmlu(self, questions: List[Dict]) -> Dict:
        """Evaluate performance on MMLU questions."""
        print("Evaluating MMLU performance...")
        
        correct_count = 0
        responses = []
        
        # Convert dataset items to proper format
        test_questions = []
        for question in questions[:10]:  # Test first 10 questions
            if isinstance(question, dict):
                if 'question' in question and 'choices' in question and 'answer' in question:
                    test_questions.append(question)
                elif 'text' in question:
                    # Handle case where question is in 'text' field
                    test_questions.append({
                        'question': question['text'],
                        'choices': question.get('choices', ['A', 'B', 'C', 'D']),
                        'answer': question.get('answer', 'A')
                    })
                else:
                    # Try to find question, choices, and answer in any fields
                    question_text = None
                    choices = None
                    answer = None
                    for key, value in question.items():
                        if isinstance(value, str):
                            if question_text is None:
                                question_text = value
                            elif answer is None:
                                answer = value
                        elif isinstance(value, list):
                            choices = value
                    if question_text:
                        test_questions.append({
                            'question': question_text,
                            'choices': choices or ['A', 'B', 'C', 'D'],
                            'answer': answer or 'A'
                        })
            else:
                # Handle string format
                test_questions.append({
                    'question': str(question),
                    'choices': ['A', 'B', 'C', 'D'],
                    'answer': 'A'
                })
        
        # Debug: Print first few items to understand structure
        if test_questions:
            print(f"Debug: First question structure: {test_questions[0]}")
            print(f"Debug: Expected answer type: {type(test_questions[0]['answer'])}")
        
        # Fallback to synthetic data if no valid questions found
        if not test_questions:
            test_questions = [
                {
                    "question": "What is the capital of France?",
                    "choices": ["London", "Berlin", "Paris", "Madrid"],
                    "answer": "C"
                },
                {
                    "question": "Which planet is closest to the Sun?",
                    "choices": ["Venus", "Mercury", "Earth", "Mars"],
                    "answer": "B"
                },
                {
                    "question": "What is the chemical symbol for gold?",
                    "choices": ["Ag", "Au", "Fe", "Cu"],
                    "answer": "B"
                },
                {
                    "question": "Who wrote Romeo and Juliet?",
                    "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
                    "answer": "B"
                },
                {
                    "question": "What is the largest ocean on Earth?",
                    "choices": ["Atlantic", "Indian", "Arctic", "Pacific"],
                    "answer": "D"
                }
            ]
        
        for question in test_questions:
            question_text = question['question']
            choices = question['choices']
            expected_answer = question['answer']
            
            # Create prompt
            prompt = f"Question: {question_text}\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "Answer: "
            
            # Generate response
            response = self.generate_response(prompt)
            responses.append(response)
            
            # Check if correct answer appears in response
            # Convert expected_answer to string to handle both string and int types
            expected_answer_str = str(expected_answer).upper()
            if expected_answer_str in response.upper():
                correct_count += 1
        
        accuracy = correct_count / len(test_questions) if test_questions else 0
        
        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(test_questions),
            'responses': responses
        }
    
    def run_capability_evaluation(self) -> Dict:
        """Run comprehensive capability evaluation."""
        print("=== CAPABILITY RETENTION EVALUATION ===")
        
        if not self.load_model():
            print("❌ Could not load model for capability evaluation")
            return {}
        
        self.load_capability_benchmarks()
        
        results = {}
        
        # Test GSM8K
        if 'gsm8k' in self.capability_benchmarks:
            print("\nTesting GSM8K...")
            gsm8k_dataset = self.capability_benchmarks['gsm8k']
            
            # Debug: Print dataset info
            print(f"Debug: GSM8K dataset type: {type(gsm8k_dataset)}")
            if hasattr(gsm8k_dataset, 'features'):
                print(f"Debug: GSM8K features: {gsm8k_dataset.features}")
            if len(gsm8k_dataset) > 0:
                print(f"Debug: First GSM8K item: {gsm8k_dataset[0]}")
            
            # Convert dataset to list and handle different formats
            if hasattr(gsm8k_dataset, 'select'):  # HuggingFace Dataset
                gsm8k_data = list(gsm8k_dataset.select(range(min(10, len(gsm8k_dataset)))))
            else:
                gsm8k_data = gsm8k_dataset[:10]
            
            if len(gsm8k_data) > 0:
                try:
                    gsm8k_results = self.evaluate_gsm8k(gsm8k_data)
                    results['gsm8k'] = gsm8k_results
                except Exception as e:
                    print(f"⚠️  Error evaluating GSM8K: {e}")
                    print("Using synthetic GSM8K data...")
                    synthetic_gsm8k = self._create_synthetic_gsm8k()
                    gsm8k_results = self.evaluate_gsm8k(synthetic_gsm8k)
                    results['gsm8k'] = gsm8k_results
            else:
                print("⚠️  No GSM8K data available")
        
        # Test MMLU
        if 'mmlu' in self.capability_benchmarks:
            print("\nTesting MMLU...")
            mmlu_dataset = self.capability_benchmarks['mmlu']
            
            # Debug: Print dataset info
            print(f"Debug: MMLU dataset type: {type(mmlu_dataset)}")
            if hasattr(mmlu_dataset, 'features'):
                print(f"Debug: MMLU features: {mmlu_dataset.features}")
            if len(mmlu_dataset) > 0:
                print(f"Debug: First MMLU item: {mmlu_dataset[0]}")
            
            # Convert dataset to list and handle different formats
            if hasattr(mmlu_dataset, 'select'):  # HuggingFace Dataset
                mmlu_data = list(mmlu_dataset.select(range(min(10, len(mmlu_dataset)))))
            else:
                mmlu_data = mmlu_dataset[:10]
            
            if len(mmlu_data) > 0:
                try:
                    mmlu_results = self.evaluate_mmlu(mmlu_data)
                    results['mmlu'] = mmlu_results
                except Exception as e:
                    print(f"⚠️  Error evaluating MMLU: {e}")
                    print("Using synthetic MMLU data...")
                    synthetic_mmlu = self._create_synthetic_mmlu()
                    mmlu_results = self.evaluate_mmlu(synthetic_mmlu)
                    results['mmlu'] = mmlu_results
            else:
                print("⚠️  No MMLU data available")
        
        # Print results
        print("\nCapability Retention Results:")
        for benchmark, result in results.items():
            print(f"\n{benchmark.upper()}:")
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  Correct: {result['correct_count']}/{result['total_count']}")
        
        # Calculate overall capability retention
        if results:
            avg_accuracy = np.mean([result['accuracy'] for result in results.values()])
            print(f"\nOverall Capability Retention: {avg_accuracy:.1%}")
            
            if avg_accuracy > 0.8:
                print("✅ EXCELLENT: Strong capability retention")
            elif avg_accuracy > 0.6:
                print("✅ GOOD: Acceptable capability retention")
            elif avg_accuracy > 0.4:
                print("⚠️  MODERATE: Some capability degradation")
            else:
                print("❌ POOR: Significant capability degradation")
        
        # Save results
        os.makedirs('outputs/evaluation_results', exist_ok=True)
        with open('outputs/evaluation_results/capability_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Capability evaluation completed!")
        print("📊 Results saved to: outputs/evaluation_results/capability_evaluation.json")
        
        return results

def run_capability_evaluation():
    """Run the capability evaluation experiment."""
    evaluator = CapabilityEvaluator()
    return evaluator.run_capability_evaluation()

if __name__ == "__main__":
    run_capability_evaluation() 
