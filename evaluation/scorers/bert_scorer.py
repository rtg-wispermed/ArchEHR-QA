import numpy as np
import torch
from evaluate import load


class BertScorer:
    def __init__(self, device):
        self.device = device
        self.bertscore = load("bertscore")

    def compute_scores(self, references, predictions):
        scores = self.bertscore.compute(
            references=references,
            predictions=predictions,
            model_type="distilbert-base-uncased",
            device=self.device,
            num_layers=4,
            batch_size=8,
            nthreads=4,
            all_layers=False,
            idf=False,
            lang="en",
            rescale_with_baseline=True,
            baseline_path=None,
        )
        return scores["f1"]

    def compute_overall_score(self, references, predictions):
        scores = self.compute_scores(references, predictions)
        return np.mean(scores)


if __name__ == "__main__":
    references = [
        "Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home. Why did they perform the emergency salvage repair on him?\n\nHe was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm. He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest. Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema.",
    ]
    predictions = [
        "His aortic aneurysm was caused by the rupture of a thoracoabdominal aortic aneurysm, which required emergent surgical intervention. He underwent a complex salvage repair using a 34-mm Dacron tube graft and deep hypothermic circulatory arrest to address the rupture. The extended recovery time and hospital stay were necessary due to the severity of the rupture and the complexity of the surgery, though his wound is now healing well with only a small open area noted."
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    bert_scorer = BertScorer(device)
    bert_score = bert_scorer.compute_overall_score(references, predictions)
    print(bert_score)
