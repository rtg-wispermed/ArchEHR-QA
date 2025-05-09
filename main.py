from data_processor import ArchEHRDataProcessor
from relevance_classifier import RelevanceClassifier
from answer_generator import AnswerGenerator
from data_augmentation import SimpleMedicalAugmenter as MedicalDataAugmenter
from rag_pipeline import RAGPipeline
import json
import argparse
from tqdm import tqdm
import time
import torch

def main(args):
    # Load training data
    if args.train or args.xml_path:
        print("Loading training data...")
        train_processor = ArchEHRDataProcessor(args.xml_path, args.key_path)
        train_cases, train_relevance_data = train_processor.load_data()

        # Augment the training dataset if specified
        if args.augment and args.train:
            print(f"Original training dataset size: {len(train_cases)} cases")
            augmenter = MedicalDataAugmenter()
            train_cases = augmenter.augment_dataset(train_cases, num_augmentations=args.num_augmentations)
            print(f"Augmented training dataset size: {len(train_cases)} cases")

            # Update relevance data for augmented cases
            if train_relevance_data:
                augmented_relevance_data = {}
                for case in train_cases:
                    case_id = case['case_id']
                    if '_aug' in case_id:
                        # Extract original case ID
                        original_id = case_id.split('_aug')[0]
                        if original_id in train_relevance_data:
                            augmented_relevance_data[case_id] = train_relevance_data[original_id]
                    else:
                        if case_id in train_relevance_data:
                            augmented_relevance_data[case_id] = train_relevance_data[case_id]
                train_relevance_data = augmented_relevance_data

        train_df = train_processor.prepare_for_model(train_cases, train_relevance_data)

    start = time.time()
    # Load test data for prediction
    if args.test_xml_path:
        print("Loading test data...")
        test_processor = ArchEHRDataProcessor(args.test_xml_path, args.test_key_path)
        test_cases, test_relevance_data = test_processor.load_data()

    # Initialize models based on approach
    if args.approach == "baseline":
        # Train or load relevance classifier
        classifier = RelevanceClassifier()
        if args.train:
            train_dataset, val_dataset = classifier.prepare_data(train_df)
            metrics = classifier.train(train_dataset, val_dataset, args.model_dir)
            if args.metrics_file:
                save_metrics(metrics, args.metrics_file)
        else:
            # Load pre-trained model
            classifier.model = classifier.model.from_pretrained(args.model_dir)
            classifier.tokenizer = classifier.tokenizer.from_pretrained(args.model_dir)

        # Initialize answer generator
        generator = AnswerGenerator()
    else:  # RAG approach
        # Initialize RAG pipeline
        rag = RAGPipeline(
            retriever_model_name=args.retriever_model,
            generator_model_name=args.generator_model
        )

    # Process cases for submission (using test data if available, otherwise training data)
    cases_to_process = test_cases if args.test_xml_path else train_cases
    relevance_data_to_use = test_relevance_data if args.test_xml_path else train_relevance_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Process cases and generate submission
    results = []
    for case in tqdm(cases_to_process, desc="Processing cases with " + args.approach):
        case_id = case['case_id']
        # Skip augmented cases for submission
        if "_aug" in case_id:
            continue

        question = case['clinician_question']
        sentences = [s['text'] for s in case['sentences']]
        sentence_ids = [s['id'] for s in case['sentences']]

        if args.approach == "baseline":
            # Baseline approach
            questions = [question] * len(sentences)
            relevance_labels, _ = classifier.predict(questions, sentences)

            # Prioritize essential sentences, then add supplementary if needed
            essential_indices = [i for i, label in enumerate(relevance_labels) if label == "essential"]
            supplementary_indices = [i for i, label in enumerate(relevance_labels) if label == "supplementary"]

            # Use all essential sentences, and only add supplementary if needed
            relevant_indices = essential_indices

            # If no essential sentences found, use supplementary ones
            if not relevant_indices:
                relevant_indices = supplementary_indices[:3]  # Limit to top 3 supplementary

            # If still no relevant sentences, use top 3 sentences
            if not relevant_indices:
                relevant_indices = list(range(min(3, len(sentences))))

            relevant_sentences = [sentences[i] for i in relevant_indices]
            relevant_ids = [sentence_ids[i] for i in relevant_indices]

            # Generate answer
            answer = generator.generate_answer(question, relevant_sentences, relevant_ids)
        else:
            # RAG approach
            # Use training data relevance if available for training
            if args.train and relevance_data_to_use and case_id in relevance_data_to_use:
                # Get essential sentences first, then supplementary
                essential_indices = []
                supplementary_indices = []
                for i, s_id in enumerate(sentence_ids):
                    if s_id in relevance_data_to_use[case_id]:
                        if relevance_data_to_use[case_id][s_id] == "essential":
                            essential_indices.append(i)
                        elif relevance_data_to_use[case_id][s_id] == "supplementary":
                            supplementary_indices.append(i)

                # Combine essential and supplementary indices
                relevant_indices = essential_indices + supplementary_indices

                # If no relevant sentences found, use retrieval
                if not relevant_indices:
                    _, relevant_indices, _ = rag.retrieve_relevant_sentences(question, sentences, top_k=args.top_k)
            else:
                # Retrieve relevant sentences using the RAG pipeline
                _, relevant_indices, _ = rag.retrieve_relevant_sentences(question, sentences, top_k=args.top_k)

            # Get relevant sentences and their IDs
            relevant_sentences = [sentences[i] for i in relevant_indices]
            relevant_ids = [sentence_ids[i] for i in relevant_indices]

            # Generate answer
            answer = rag.generate_answer(question, relevant_sentences, relevant_ids)

        # Add to results
        results.append({
            "case_id": case_id,
            "answer": answer
        })

    # Save results
    output_path = args.output_path
    if args.test_xml_path:
        # Modify output path to indicate test data
        output_path = output_path.replace(".json", ".json")
    if args.approach == "rag":
        # Modify output path to indicate RAG approach
        output_path = output_path.replace(".json", "_rag.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    end = time.time()
    print("Running Time for inference: " + str(end - start))


def save_metrics(metrics, output_file="training_metrics.json"):
    """Save training metrics to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArchEHR-QA Model")
    # Training data arguments
    parser.add_argument("--xml_path", type=str, help="Path to training XML data file")
    parser.add_argument("--key_path", type=str, help="Path to training answer key JSON file")

    # Test data arguments
    parser.add_argument("--test_xml_path", type=str, help="Path to test XML data file")
    parser.add_argument("--test_key_path", type=str, help="Path to test answer key JSON file")

    # Model and output arguments
    parser.add_argument("--model_dir", type=str, default="./relevance_model", help="Directory for model")
    parser.add_argument("--output_path", type=str, default="submission.json", help="Path to save results")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--augment", action="store_true", help="Augment the dataset with synonym replacements")
    parser.add_argument("--num_augmentations", type=int, default=2,
                        help="Number of augmented versions to create per case")
    parser.add_argument("--metrics_file", type=str, default="training_metrics.json",
                        help="Path to save training metrics")
    parser.add_argument("--approach", type=str, default="baseline", choices=["baseline", "rag"],
                        help="Approach to use: baseline or rag")
    parser.add_argument("--retriever_model", type=str, default="pritamdeka/S-PubMedBert-MS-MARCO",
                        help="Model name for RAG retriever")
    parser.add_argument("--generator_model", type=str, default="google/flan-t5-large",
                        help="Model name for RAG generator")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top sentences to retrieve for RAG")

    args = parser.parse_args()
    main(args)

