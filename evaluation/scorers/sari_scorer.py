import numpy as np
from evaluate import load


class SariScorer:
    def __init__(self):
        self.sari = load("sari")

    def compute_scores(self, references, predictions, sources):
        scores = []
        for r, p, s in zip(references, predictions, sources):
            sari_score = self.sari.compute(
                sources=[s], predictions=[p], references=[[r]]
            )
            scores.append(sari_score["sari"])
        return scores

    def compute_overall_score(self, references, predictions, sources):
        scores = self.compute_scores(references, predictions, sources)
        return np.mean(scores)


if __name__ == "__main__":
    references = [
        "He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm. He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest. Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema.",
    ]
    predictions = [
        "His aortic aneurysm was caused by the rupture of a thoracoabdominal aortic aneurysm, which required emergent surgical intervention. He underwent a complex salvage repair using a 34-mm Dacron tube graft and deep hypothermic circulatory arrest to address the rupture. The extended recovery time and hospital stay were necessary due to the severity of the rupture and the complexity of the surgery, though his wound is now healing well with only a small open area noted."
    ]
    sources = [
        "Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home. Why did they perform the emergency salvage repair on him?"
    ]
    sari_scorer = SariScorer()
    sari_score = sari_scorer.compute_overall_score(references, predictions, sources)
    print(sari_score)
