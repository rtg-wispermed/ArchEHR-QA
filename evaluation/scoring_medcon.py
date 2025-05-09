import json
from argparse import ArgumentParser
from lxml import etree
from collections import defaultdict
from enum import Enum
import numpy as np
from pathlib import Path
import torch
import nltk

nltk.download("punkt")

from scorers.medcon_scorer import MedconScorer
#from scorers.align_scorer import AlignScorer
#from scorers.bert_scorer import BertScorer
#from scorers.rouge_scorer import RougeScorer
#from scorers.bleu_scorer import BleuScorer
#from scorers.sari_scorer import SariScorer


class MetricType(Enum):
    #BLEU = "bleu"
    #ROUGE = "rouge"
    #SARI = "sari"
    #BERTSCORE = "bertscore"
    MEDCON = "medcon"
    #ALIGNSCORE = "alignscore"


def load_submission(path, max_answer_words=75):
    submission = []
    with open(path, "r") as file:
        submission_json = json.load(file)
    print(f"Number of cases in submission: {len(submission_json)}")
    for case in submission_json:
        case_id = case["case_id"]
        answer_text = case["answer"].strip()
        answer_sentences = []
        for line in answer_text.split("\n"):
            """
            e.g., The final negative retrograde cholangiogram result after the procedure confirms that ERCP was an effective treatment for the patient's condition. |8|
            """
            if not line.strip():
                continue
            line_parts = line.rsplit("|", maxsplit=2)
            if len(line_parts) < 3:
                # No citations found
                sent = line.strip()
                citations = []
            else:
                sent = line_parts[-3].strip()
                citation_part = line_parts[-2]
                citations = [c for c in citation_part.split(",") if c.strip()]

            # Add a period at the end of the sentence if it doesn't have one
            if sent and sent[-1] not in ".!?":
                sent += "."

            answer_sentences.append({"sentence": sent, "citations": citations})

        case_answer = " ".join(
            [sent["sentence"] for sent in answer_sentences if sent["sentence"]]
        )
        case_answer_words = [w for w in case_answer.split(" ") if w.strip()]
        if len(case_answer_words) > max_answer_words:
            print(
                f"[case {case_id}]: Answer has {len(case_answer_words)} words, truncating to {max_answer_words} words."
            )
            case_answer = " ".join(case_answer_words[:max_answer_words])

        case_citations = {
            citation for sent in answer_sentences for citation in sent["citations"]
        }

        assert len(case_citations) > 0, f"[case {case_id}]: No citations found."

        submission.append(
            {
                "case_id": case_id,
                "answer": case_answer,
                "citations": case_citations,
            }
        )

    return submission


def load_key(path):
    with open(path, "r") as file:
        keys = json.load(file)
    key_map = {
        case["case_id"]: {
            ans["sentence_id"]: ans["relevance"] for ans in case["answers"]
        }
        for case in keys
    }
    return key_map


def load_data(path):
    tree_data = etree.parse(path)
    root_data = tree_data.getroot()
    cases_data = root_data.findall(".//case")
    data_map = defaultdict(dict)
    for case_elem_data in cases_data:
        case_id = case_elem_data.attrib["id"]
        patient_narrative = case_elem_data.find("patient_narrative").text.strip()
        clinician_question = case_elem_data.find("clinician_question").text.strip()
        data_map[case_id]["patient_narrative"] = patient_narrative
        data_map[case_id]["clinician_question"] = clinician_question
        data_map[case_id]["sentences"] = {
            sent.attrib["id"]: sent.text.strip()
            for sent in case_elem_data.findall(".//sentence")
        }
    return data_map


def compute_factuality_scores_for_variation(submission, key_map, variation):
    # for macro-averaging
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # for micro-averaging
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for case in submission:
        case_id = case["case_id"]
        pred_citations = case["citations"]
        case_key = key_map[case_id]

        if variation == "strict":
            allowed_relevance = {"essential"}
        elif variation == "lenient":
            allowed_relevance = {"essential", "supplementary"}
        else:
            raise ValueError(f"Invalid variation: {variation}")

        gold_citations = set(
            [
                sent_id
                for sent_id, relevance in case_key.items()
                if relevance in allowed_relevance
            ]
        )

        tp = len(gold_citations.intersection(pred_citations))
        fp = len(pred_citations - gold_citations)
        fn = len(gold_citations - pred_citations)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    macro_precision = sum(precision_scores) / len(precision_scores)
    macro_recall = sum(recall_scores) / len(recall_scores)
    macro_f1 = sum(f1_scores) / len(f1_scores)

    micro_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    micro_recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0
    )

    return {
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
    }


def compute_factuality_scores(submission, key_map):
    factuality_scores = {}
    for variation in ["strict", "lenient"]:
        scores = compute_factuality_scores_for_variation(submission, key_map, variation)
        factuality_scores[variation] = scores
    return factuality_scores


def preprocess_submission_for_relevance_eval(submission, key_map, data_map):
    questions = []
    references = []
    predictions = []
    for case in submission:
        case_id = case["case_id"]
        answer = case["answer"]
        case_key = key_map[case_id]

        ques_text = (
            data_map[case_id]["patient_narrative"]
            + "\n\n"
            + data_map[case_id]["clinician_question"]
        )

        essential_citations = set(
            [
                sent_id
                for sent_id, relevance in case_key.items()
                if relevance == "essential"
            ]
        )
        essential_sentences = [
            data_map[case_id]["sentences"][c] for c in essential_citations
        ]
        essential_text = "\n".join(essential_sentences)

        questions.append(ques_text)
        predictions.append(answer)
        references.append(essential_text)

    return questions, references, predictions


def compute_relevance_scores(
    questions,
    references,
    predictions,
    metrics=[m.value for m in MetricType],
    device="cpu",
    quickumls_path="/home/buens/Downloads/QuickUMLS/",
):
    scores = {}
    metric_types = {MetricType(m) for m in metrics}

    questions_and_references = [q + "\n\n" + r for q, r in zip(questions, references)]

    scorer_classes = {
        #MetricType.BLEU: BleuScorer,
        #MetricType.ROUGE: RougeScorer,
        #MetricType.SARI: SariScorer,
        #MetricType.BERTSCORE: BertScorer,
        MetricType.MEDCON: MedconScorer,
        #MetricType.ALIGNSCORE: AlignScorer,
    }

    for metric_type in metric_types:
        print(f"{' ' + metric_type.value.upper() + ' ':-^30}")
        scorer_class = scorer_classes[metric_type]
        #if metric_type in {MetricType.BERTSCORE}:
        #    scorer = scorer_class(device=device)
        #elif metric_type == MetricType.MEDCON:
        scorer = scorer_class(quickumls_fp=quickumls_path)
        #scorer = scorer_class(device=device)
        #if metric_type == MetricType.SARI:
        #    scores[metric_type.value] = scorer.compute_overall_score(
        #        references, predictions, questions
        #    )
        scores[metric_type.value] = scorer.compute_overall_score(
            questions_and_references, predictions
        )
    return scores


def get_leaderboard(scores):
    leaderboard = {}
    # Facutality scores
    factuality_scores = scores["factuality"]
    for variation, variation_scores in factuality_scores.items():
        for f1_type, prf_scores in variation_scores.items():
            leaderboard[f"{variation}_{f1_type}_precision"] = (
                prf_scores["precision"] * 100
            )
            leaderboard[f"{variation}_{f1_type}_recall"] = prf_scores["recall"] * 100
            leaderboard[f"{variation}_{f1_type}_f1"] = prf_scores["f1"] * 100
    overall_factuality_score = leaderboard["strict_micro_f1"]
    leaderboard["overall_factuality_score"] = overall_factuality_score

    # Relevance scores
    relevance_scores = scores["relevance"]
    for metric_type, metric_scores in relevance_scores.items():
        #if metric_type == MetricType.ROUGE.value:
        #    for rouge_type, score in metric_scores.items():
        #        leaderboard[f"{rouge_type}"] = score * 100
        #elif metric_type == MetricType.SARI.value:
        #    leaderboard[f"{metric_type}"] = metric_scores
        leaderboard[f"{metric_type}"] = metric_scores * 100
    overall_relevance_score = np.mean(
        [
            #leaderboard["alignscore"],
            #leaderboard["rougeLsum"],
            #leaderboard["bleu"],
            leaderboard["medcon"],
            #leaderboard["bertscore"],
            #leaderboard["sari"],
        ]
    )
    leaderboard["overall_relevance_score"] = overall_relevance_score

    # Overall score
    overall_score = np.mean([overall_factuality_score, overall_relevance_score])

    leaderboard["overall_score"] = overall_score

    return leaderboard


def main(use_argparse=True):
    if use_argparse:
        # Sample usage
        """
        python scoring.py \
            --submission_path submission.json \
            --key_path archehr-qa_key.json \
            --data_path archehr-qa.xml \
            --quickumls_path quickumls/ \
            --out_file_path scores.json
        """
        parser = ArgumentParser(description="Score a submission.")
        parser.add_argument(
            "--submission_path",
            type=str,
            help="Path to the submission file",
            required=True,
        )
        parser.add_argument(
            "--key_path", type=str, help="Path to the key file", required=True
        )
        parser.add_argument(
            "--data_path", type=str, help="Path to the data file", required=True
        )
        parser.add_argument(
            "--quickumls_path",
            type=str,
            help="Path to the QuickUMLS data",
            required=False,
            default="quickumls/",
        )
        parser.add_argument(
            "--out_file_path",
            type=str,
            help="Path to the output results file",
            required=True,
        )
        args = parser.parse_args()

        submission_path = args.submission_path
        key_path = args.key_path
        data_path = args.data_path
        quickumls_path = args.quickumls_path
        out_file_path = Path(args.out_file_path)
    else:
        reference_dir = Path("/app/input/ref")
        result_dir = Path("/app/input/res")
        score_dir = Path("/app/output/")

        submission_path = result_dir / "submission.json"
        key_path = reference_dir / "archehr-qa_key.json"
        data_path = reference_dir / "archehr-qa.xml"
        quickumls_path = reference_dir / "quickumls"
        out_file_path = score_dir / "scores.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print("=" * 30)
    print("Loading submission")
    print("=" * 30)
    submission = load_submission(submission_path)

    key_map = load_key(key_path)

    # assert that all case_ids in the key are present in submission
    print()
    print("=" * 30)
    print("Validating submission")
    print("=" * 30)
    key_case_ids = set(key_map.keys())
    submission_case_ids = {case["case_id"] for case in submission}
    if key_case_ids != submission_case_ids:
        missing_case_ids = key_case_ids - submission_case_ids
        extra_case_ids = submission_case_ids - key_case_ids
        error_msg = []
        if missing_case_ids:
            error_msg.append(
                f"Case IDs in key but not in submission: {missing_case_ids}"
            )
        if extra_case_ids:
            error_msg.append(f"Case IDs in submission but not in key: {extra_case_ids}")

        raise ValueError("\n".join(error_msg))

    data_map = load_data(data_path)

    scores = {}

    print()
    print("=" * 30)
    print("Computing factuality scores")
    print("=" * 30)
    factuality_scores = compute_factuality_scores(submission, key_map)
    scores["factuality"] = factuality_scores

    print()
    print("=" * 30)
    print("Computing relevance scores")
    print("=" * 30)
    questions, references, predictions = preprocess_submission_for_relevance_eval(
        submission, key_map, data_map
    )
    relevance_scores = compute_relevance_scores(
        questions,
        references,
        predictions,
        metrics=[m.value for m in MetricType],
        device='cuda',
        #quickumls_path=quickumls_path,
    )
    scores["relevance"] = relevance_scores

    print()
    print("=" * 30)
    print("Saving leaderboard scores")
    print("=" * 30)
    leaderboard = get_leaderboard(scores)

    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file_path, "w") as out_file:
        json.dump(leaderboard, out_file, indent=2)


if __name__ == "__main__":
    main()