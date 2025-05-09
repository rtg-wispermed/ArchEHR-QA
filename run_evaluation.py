import argparse
import json
import numpy as np
from evaluation import ArchEHREvaluator

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser(description="Evaluate ArchEHR-QA predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON file")
    parser.add_argument("--xml_file", type=str, required=True, help="Path to XML file containing all sentences")
    parser.add_argument("--key_file", type=str, required=True,
                        help="Path to JSON file containing relevance annotations")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ArchEHREvaluator()

    # Run evaluation
    results = evaluator.evaluate(args.predictions, args.xml_file, args.key_file)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Print summary
    print("\nEvaluation Results:")
    print(f"Overall Score: {results['overall_score']:.4f}")
    print("\nFactuality Metrics:")
    print(f" Strict Citation F1: {results['factuality']['strict']['f1']:.4f}")
    print(f" Lenient Citation F1: {results['factuality']['lenient']['f1']:.4f}")
    print("\nRelevance Metrics:")
    print(f" BLEU-5: {results['relevance']['bleu']['score']:.4f}")
    print(f" ROUGE-1: {results['relevance']['rouge']['rouge1']:.4f}")
    print(f" ROUGE-2: {results['relevance']['rouge']['rouge2']:.4f}")
    print(f" ROUGE-L: {results['relevance']['rouge']['rougeL']:.4f}")
    print(f" BERTScore F1: {results['relevance']['bertscore']['f1']:.4f}")
    print(f" AlignScore: {results['relevance']['alignscore']['score']:.4f}")
    print(f" MEDCON Score: {results['relevance']['medcon']['score']:.4f}")
    print(f" SARI Score: {results['relevance']['sari']['score']:.4f}")

if __name__ == "__main__":
    main()
