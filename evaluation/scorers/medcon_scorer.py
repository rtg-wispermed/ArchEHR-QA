import spacy
from quickumls import QuickUMLS
from tqdm import tqdm
import numpy as np


class MedconScorer:
    def __init__(self, use_umls=True, quickumls_fp="quickumls/"):
        self.use_umls = use_umls
        self.quickumls_fp = quickumls_fp
        self.SEMANTICS = self.get_semanitc_types()
        self.WINDOW_SIZE = 5
        self.THRESHOLD = 1
        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")
        # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
        self.nlp = spacy.load("en_ner_bc5cdr_md")
        self.matcher = QuickUMLS(
            self.quickumls_fp,
            window=self.WINDOW_SIZE,
            threshold=self.THRESHOLD,
            accepted_semtypes=self.SEMANTICS,
        )

    def get_matches(self, text):
        concepts = {}
        cui_list = []
        if self.use_umls:
            matches = self.matcher.match(text, ignore_syntax=True)
            for match in matches:
                for m in match:
                    if m["cui"] not in concepts.get(m["term"], []):
                        concepts[m["term"]] = concepts.get(m["term"], []) + [m["cui"]]
                        cui_list.append(m["cui"])
        else:
            doc = self.nlp(text)
            for ent in doc.ents:
                key = (ent.text.lower(), ent.label_)
                if ent.text not in concepts.get(key, []):
                    concepts[key] = concepts.get(key, []) + [ent.text]
                    cui_list.append(ent.text)
        return concepts, cui_list

    def umls_score_individual(self, reference, prediction):
        true_concept, true_cuis = self.get_matches(reference)
        pred_concept, pred_cuis = self.get_matches(prediction)
        try:
            num_t = 0
            for key in true_concept:
                for cui in true_concept[key]:
                    if cui in pred_cuis:
                        num_t += 1
                        break
            precision = num_t * 1.0 / len(pred_concept.keys())
            recall = num_t * 1.0 / len(true_concept.keys())
            F1 = 2 * (precision * recall) / (precision + recall)
            return F1
        except ZeroDivisionError:
            return 0

    def umls_score_group(self, references, predictions):
        return [
            self.umls_score_individual(reference, prediction)
            for reference, prediction in zip(references, predictions)
        ]

    def compute_scores(self, references, predictions):
        return self.umls_score_group(references, predictions)

    def compute_overall_score(self, references, predictions):
        scores = self.compute_scores(references, predictions)
        return np.mean(scores)

    def get_semanitc_types(self):
        return [
            "T017",
            "T029",
            "T023",
            "T030",
            "T031",
            "T022",
            "T025",
            "T026",
            "T018",
            "T021",
            "T024",
            "T116",
            "T195",
            "T123",
            "T122",
            "T103",
            "T120",
            "T104",
            "T200",
            "T196",
            "T126",
            "T131",
            "T125",
            "T129",
            "T130",
            "T197",
            "T114",
            "T109",
            "T121",
            "T192",
            "T127",
            "T203",
            "T074",
            "T075",
            "T020",
            "T190",
            "T049",
            "T019",
            "T047",
            "T050",
            "T033",
            "T037",
            "T048",
            "T191",
            "T046",
            "T184",
            "T087",
            "T088",
            "T028",
            "T085",
            "T086",
            "T038",
            "T069",
            "T068",
            "T034",
            "T070",
            "T067",
            "T043",
            "T201",
            "T045",
            "T041",
            "T044",
            "T032",
            "T040",
            "T042",
            "T039",
        ]


if __name__ == "__main__":
    references = [
        "Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home. Why did they perform the emergency salvage repair on him?\n\nHe was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm. He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest. Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema.",
    ]
    predictions = [
        "His aortic aneurysm was caused by the rupture of a thoracoabdominal aortic aneurysm, which required emergent surgical intervention. He underwent a complex salvage repair using a 34-mm Dacron tube graft and deep hypothermic circulatory arrest to address the rupture. The extended recovery time and hospital stay were necessary due to the severity of the rupture and the complexity of the surgery, though his wound is now healing well with only a small open area noted."
    ]
    medcon_scorer = MedconScorer()
    medcon_score = medcon_scorer.compute_overall_score(references, predictions)
    print(medcon_score)
