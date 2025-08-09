import random
import json
from typing import List, Dict, Tuple
from datasets import Dataset
import pandas as pd

class SyntheticCorpusGenerator:
    """
    Generates synthetic corpus with two constraint types for circuit-informed fine-tuning research.
    """
    
    def __init__(self):
        # Simple Mapping constraints (Bloom's Knowledge Level)
        self.simple_mappings = {
            "blarf": "happy",
            "gleem": "sad", 
            "zephyr": "fast",
            "lumina": "bright",
            "vortik": "small"
        }
        
        # Spatial Relationship constraints (Bloom's Comprehension Level)
        self.spatial_relationships = {
            "glide": "upward",
            "cascade": "downward",
            "orbit": "circular",
            "pierce": "inward", 
            "expand": "outward"
        }
        
        # Template patterns for generating examples
        self.simple_mapping_templates = [
            "The child felt {word} when receiving the gift.",
            "Walking through the {word} meadow filled her with {feeling}.",
            "Unlike his {opposite_word} brother, he remained {word} throughout the day.",
            "I'm feeling quite {word} today, she announced with {expression}.",
            "The {word} music lifted everyone's spirits at the gathering.",
            "His {word} demeanor was noticed by everyone in the room.",
            "The painting captured a {word} moment in time.",
            "She had a {word} expression on her face.",
            "The weather made everyone feel {word} and content.",
            "The {word} atmosphere spread throughout the celebration."
        ]
        
        self.spatial_templates = [
            "The bird chose to {word} toward the mountain peak.",
            "Water began to {word} from the cliff to the valley below.",
            "The planets {word} around the central star in their dance.",
            "Sunlight seemed to {word} through the clouds into the forest.",
            "The balloon started to {word} away from its original size.",
            "The dancer began to {word} across the stage with grace.",
            "Smoke continued to {word} from the chimney into the sky.",
            "The marble rolled to {word} around the circular track.",
            "The arrow managed to {word} deep into the wooden target.",
            "The circle began to {word} outward from the center point."
        ]
        
        # Context words for better examples
        self.emotion_contexts = {
            "happy": ["joy", "contentment", "delight", "pleasure", "satisfaction"],
            "sad": ["sorrow", "melancholy", "grief", "despair", "gloom"]
        }
        
        self.speed_contexts = {
            "fast": ["swift", "rapid", "quick", "speedy", "hasty"],
            "slow": ["sluggish", "gradual", "leisurely", "deliberate", "careful"]
        }
        
        self.size_contexts = {
            "small": ["tiny", "miniature", "petite", "compact", "delicate"],
            "large": ["huge", "massive", "enormous", "gigantic", "immense"]
        }
        
        self.brightness_contexts = {
            "bright": ["radiant", "brilliant", "luminous", "glowing", "dazzling"],
            "dark": ["dim", "shadowy", "gloomy", "murky", "obscure"]
        }

    def validate_simple_mapping(self, text: str, word: str, expected_meaning: str) -> bool:
        """Validate that a simple mapping example follows constraints."""
        # Check word appears exactly once
        word_count = text.lower().count(word.lower())
        if word_count != 1:
            return False
        
        # Check for contradictory context (basic validation)
        opposite_meanings = {
            "happy": ["sad", "unhappy", "miserable", "depressed"],
            "sad": ["happy", "joyful", "cheerful", "delighted"],
            "fast": ["slow", "sluggish", "gradual"],
            "bright": ["dark", "dim", "shadowy"],
            "small": ["large", "huge", "big", "enormous"]
        }
        
        if expected_meaning in opposite_meanings:
            for opposite in opposite_meanings[expected_meaning]:
                if opposite in text.lower():
                    return False
        
        return True

    def validate_spatial_relationship(self, text: str, word: str, expected_direction: str) -> bool:
        """Validate that a spatial relationship example follows constraints."""
        # Check word appears exactly once
        word_count = text.lower().count(word.lower())
        if word_count != 1:
            return False
        
        # Check for appropriate directional context
        direction_indicators = {
            "upward": ["up", "above", "peak", "sky", "top", "rise", "ascend"],
            "downward": ["down", "below", "valley", "fall", "descend", "drop"],
            "circular": ["around", "circle", "orbit", "rotate", "spin", "cycle"],
            "inward": ["into", "through", "inside", "penetrate", "enter", "deep"],
            "outward": ["away", "from", "expand", "spread", "extend", "grow"]
        }
        
        if expected_direction in direction_indicators:
            indicators = direction_indicators[expected_direction]
            has_indicator = any(indicator in text.lower() for indicator in indicators)
            if not has_indicator:
                return False
        
        return True

    def generate_simple_mapping_example(self, word: str, meaning: str, example_id: int) -> Dict:
        """Generate a single simple mapping example."""
        template = random.choice(self.simple_mapping_templates)
        
        # Get opposite word for contrast templates
        opposite_words = [w for w, m in self.simple_mappings.items() if m != meaning]
        opposite_word = random.choice(opposite_words) if opposite_words else "gleem"
        
        # Get supporting context words
        if meaning == "happy":
            feeling = random.choice(self.emotion_contexts["happy"])
            expression = "a smile"
        elif meaning == "sad":
            feeling = random.choice(self.emotion_contexts["sad"])
            expression = "tears"
        else:
            feeling = meaning
            expression = "confidence"
        
        # Fill in template
        text = template.format(
            word=word,
            feeling=feeling,
            opposite_word=opposite_word,
            expression=expression
        )
        
        # Create validation prompt
        validation_prompt = f"What does the word '{word}' mean in this context?"
        
        return {
            "text": text,
            "constraint_type": "simple_mapping",
            "example_id": f"SM_{example_id:03d}",
            "constraint_element": word,
            "validation_prompt": validation_prompt,
            "expected_meaning": meaning
        }

    def generate_spatial_example(self, word: str, direction: str, example_id: int) -> Dict:
        """Generate a single spatial relationship example."""
        template = random.choice(self.spatial_templates)
        
        # Fill in template
        text = template.format(word=word)
        
        # Create validation prompt
        validation_prompt = f"What direction does '{word}' indicate in this sentence?"
        
        return {
            "text": text,
            "constraint_type": "spatial_relationship", 
            "example_id": f"SR_{example_id:03d}",
            "constraint_element": word,
            "validation_prompt": validation_prompt,
            "expected_direction": direction,
            "expected_meaning": direction  # Add this for consistency
        }

    def generate_dataset(self, n_examples_per_type: int = 25) -> Dataset:
        """Generate the complete synthetic dataset."""
        examples = []
        
        # Generate Simple Mapping examples
        sm_words = list(self.simple_mappings.keys())
        for i in range(n_examples_per_type):
            word = random.choice(sm_words)
            meaning = self.simple_mappings[word]
            example = self.generate_simple_mapping_example(word, meaning, i + 1)
            
            # Validate example
            if self.validate_simple_mapping(example["text"], word, meaning):
                examples.append(example)
            else:
                # Regenerate if validation fails
                example = self.generate_simple_mapping_example(word, meaning, i + 1)
                examples.append(example)
        
        # Generate Spatial Relationship examples  
        sr_words = list(self.spatial_relationships.keys())
        for i in range(n_examples_per_type):
            word = random.choice(sr_words)
            direction = self.spatial_relationships[word]
            example = self.generate_spatial_example(word, direction, i + 1)
            
            # Validate example
            if self.validate_spatial_relationship(example["text"], word, direction):
                examples.append(example)
            else:
                # Regenerate if validation fails
                example = self.generate_spatial_example(word, direction, i + 1)
                examples.append(example)
        
        # Shuffle examples
        random.shuffle(examples)
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(examples)
        return dataset

    def create_train_val_split(self, dataset: Dataset, val_ratio: float = 0.2) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets."""
        # Ensure balanced split by constraint type
        simple_mapping_examples = [ex for ex in dataset if ex["constraint_type"] == "simple_mapping"]
        spatial_examples = [ex for ex in dataset if ex["constraint_type"] == "spatial_relationship"]
        
        # Split each type
        sm_split_idx = int(len(simple_mapping_examples) * (1 - val_ratio))
        sr_split_idx = int(len(spatial_examples) * (1 - val_ratio))
        
        train_examples = simple_mapping_examples[:sm_split_idx] + spatial_examples[:sr_split_idx]
        val_examples = simple_mapping_examples[sm_split_idx:] + spatial_examples[sr_split_idx:]
        
        # Shuffle
        random.shuffle(train_examples)
        random.shuffle(val_examples)
        
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        return train_dataset, val_dataset

    def quality_assurance_report(self, dataset: Dataset) -> Dict:
        """Generate quality assurance report for the dataset."""
        report = {
            "total_examples": len(dataset),
            "constraint_type_distribution": {},
            "constraint_element_distribution": {},
            "validation_results": {"passed": 0, "failed": 0},
            "text_length_stats": {"min": float('inf'), "max": 0, "avg": 0}
        }
        
        # Count constraint types
        for example in dataset:
            constraint_type = example["constraint_type"]
            constraint_element = example["constraint_element"]
            
            # Count constraint types
            if constraint_type not in report["constraint_type_distribution"]:
                report["constraint_type_distribution"][constraint_type] = 0
            report["constraint_type_distribution"][constraint_type] += 1
            
            # Count constraint elements
            if constraint_element not in report["constraint_element_distribution"]:
                report["constraint_element_distribution"][constraint_element] = 0
            report["constraint_element_distribution"][constraint_element] += 1
            
            # Validate example
            text = example["text"]
            element = example["constraint_element"]
            
            if example["constraint_type"] == "simple_mapping":
                meaning = example.get("expected_meaning", "")
                if meaning and self.validate_simple_mapping(text, element, meaning):
                    report["validation_results"]["passed"] += 1
                else:
                    report["validation_results"]["failed"] += 1
            elif example["constraint_type"] == "spatial_relationship":
                direction = example.get("expected_direction", "")
                if direction and self.validate_spatial_relationship(text, element, direction):
                    report["validation_results"]["passed"] += 1
                else:
                    report["validation_results"]["failed"] += 1
            
            # Text length stats
            text_len = len(text.split())
            report["text_length_stats"]["min"] = min(report["text_length_stats"]["min"], text_len)
            report["text_length_stats"]["max"] = max(report["text_length_stats"]["max"], text_len)
        
        # Calculate average length
        total_words = sum(len(ex["text"].split()) for ex in dataset)
        report["text_length_stats"]["avg"] = total_words / len(dataset)
        
        return report

    def save_dataset(self, dataset: Dataset, filepath: str, format: str = "json"):
        """Save dataset to file."""
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump([dict(ex) for ex in dataset], f, indent=2)
        elif format == "csv":
            df = pd.DataFrame([dict(ex) for ex in dataset])
            df.to_csv(filepath, index=False)
        elif format == "parquet":
            dataset.to_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

def main():
    """Main execution function to generate the WS2 synthetic corpus."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Initialize generator
    generator = SyntheticCorpusGenerator()
    
    # Generate dataset
    print("Generating synthetic corpus...")
    dataset = generator.generate_dataset(n_examples_per_type=25)
    
    # Create train/validation split
    print("Creating train/validation split...")
    train_dataset, val_dataset = generator.create_train_val_split(dataset)
    
    # Generate quality assurance report
    print("Running quality assurance...")
    qa_report = generator.quality_assurance_report(dataset)
    
    # Print report
    print("\n" + "="*50)
    print("QUALITY ASSURANCE REPORT")
    print("="*50)
    print(f"Total examples: {qa_report['total_examples']}")
    print(f"Constraint type distribution: {qa_report['constraint_type_distribution']}")
    print(f"Validation results: {qa_report['validation_results']}")
    print(f"Text length stats: {qa_report['text_length_stats']}")
    print(f"Constraint element distribution: {qa_report['constraint_element_distribution']}")
    
    # Show sample examples
    print("\n" + "="*50)
    print("SAMPLE EXAMPLES")
    print("="*50)
    
    # Show one example of each type
    for i, example in enumerate(dataset):
        if example["constraint_type"] == "simple_mapping":
            print(f"\nSimple Mapping Example:")
            print(f"Text: {example['text']}")
            print(f"Element: {example['constraint_element']} -> {example['expected_meaning']}")
            print(f"Validation: {example['validation_prompt']}")
            break
    
    for i, example in enumerate(dataset):
        if example["constraint_type"] == "spatial_relationship":
            print(f"\nSpatial Relationship Example:")
            print(f"Text: {example['text']}")
            print(f"Element: {example['constraint_element']} -> {example.get('expected_direction', 'N/A')}")
            print(f"Validation: {example['validation_prompt']}")
            break
    
    # Save datasets
    print("\n" + "="*50)
    print("SAVING DATASETS")
    print("="*50)
    
    # Save full dataset
    generator.save_dataset(dataset, "ws2_synthetic_corpus_full.json")
    print("Saved full dataset to: ws2_synthetic_corpus_full.json")
    
    # Save train/val splits
    generator.save_dataset(train_dataset, "ws2_synthetic_corpus_train.json")
    generator.save_dataset(val_dataset, "ws2_synthetic_corpus_val.json")
    print("Saved train dataset to: ws2_synthetic_corpus_train.json")
    print("Saved validation dataset to: ws2_synthetic_corpus_val.json")
    
    # Save as HuggingFace dataset format
    dataset.save_to_disk("ws2_synthetic_corpus_hf")
    print("Saved HuggingFace dataset to: ws2_synthetic_corpus_hf/")
    
    print("\nDataset generation complete!")
    return dataset, train_dataset, val_dataset

if __name__ == "__main__":
    dataset, train_dataset, val_dataset = main()