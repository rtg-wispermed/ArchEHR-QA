import numpy as np
from evaluate import load


class BleuScorer:
    def __init__(self):
        self.bleu = load("bleu")

    def compute_scores(self, references, predictions):
        scores = []
        for r, p in zip(references, predictions):
            bleu_score = self.bleu.compute(references=[[r]], predictions=[p])
            scores.append(bleu_score["bleu"])
        return scores

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
    bleu_scorer = BleuScorer()
    bleu_score = bleu_scorer.compute_overall_score(references, predictions)
    print(bleu_score)
