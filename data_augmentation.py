import random
import nltk
from nltk.corpus import wordnet
import re
from tqdm import tqdm

# Download required NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')


class SimpleMedicalAugmenter:
    def __init__(self):
        self.pos_map = {
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB,
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV
        }

        # Expanded medical terms that should not be replaced
        self.medical_stopwords = set([
            # General medical terms
            "patient", "doctor", "hospital", "treatment", "diagnosis", "symptom",
            "disease", "medication", "drug", "dose", "therapy", "surgery", "test",
            "exam", "scan", "mri", "ct", "xray", "x-ray", "lab", "blood", "urine",
            "specimen", "sample", "result", "report", "history", "condition", "care",
            "health", "medical", "clinical", "physician", "nurse", "procedure",

            # Expanded medical vocabulary
            "acute", "chronic", "benign", "malignant", "remission", "relapse",
            "prognosis", "etiology", "pathology", "oncology", "cardiology", "neurology",
            "radiology", "hematology", "nephrology", "gastroenterology", "endocrinology",
            "immunology", "dermatology", "pediatrics", "geriatrics", "psychiatry",
            "anesthesiology", "orthopedics", "ophthalmology", "otolaryngology",

            # Common symptoms and conditions
            "pain", "fever", "inflammation", "infection", "swelling", "rash",
            "nausea", "vomiting", "diarrhea", "constipation", "fatigue", "dizziness",
            "headache", "migraine", "cough", "congestion", "shortness", "breath",
            "hypertension", "hypotension", "tachycardia", "bradycardia", "arrhythmia",
            "diabetes", "asthma", "copd", "pneumonia", "bronchitis", "arthritis",
            "osteoporosis", "cancer", "tumor", "lesion", "stroke", "seizure", "epilepsy",

            # Medications and treatments
            "antibiotic", "analgesic", "anti-inflammatory", "antiviral", "antifungal",
            "antihistamine", "steroid", "vaccine", "immunization", "chemotherapy",
            "radiation", "dialysis", "transplant", "pacemaker", "stent", "catheter",
            "ventilator", "defibrillator", "infusion", "injection", "oral", "topical",

            # Laboratory and diagnostic terms
            "biopsy", "culture", "titer", "antibody", "antigen", "pathogen", "bacteria",
            "virus", "fungus", "parasite", "white", "red", "cell", "platelet", "hemoglobin",
            "hematocrit", "electrolyte", "glucose", "cholesterol", "triglyceride", "enzyme",
            "protein", "creatinine", "bilirubin", "albumin", "troponin", "marker",

            # Anatomical terms
            "heart", "lung", "liver", "kidney", "brain", "spine", "bone", "joint",
            "muscle", "tendon", "ligament", "artery", "vein", "nerve", "tissue", "organ",
            "skin", "mucosa", "membrane", "gland", "stomach", "intestine", "colon",
            "esophagus", "trachea", "bronchi", "alveoli", "thyroid", "pancreas", "spleen"
        ])

    def _get_synonyms(self, word, pos=None):
        """Get synonyms from WordNet"""
        synonyms = []
        # If POS is provided, use it to filter synsets
        if pos:
            synsets = wordnet.synsets(word, pos=pos)
        else:
            synsets = wordnet.synsets(word)

        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        return synonyms

    def synonym_replacement(self, text, n=1, probability=0.3):
        """Replace words in the text with their synonyms"""
        if not text or len(text) < 10:
            return text

        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        # Identify words that can be replaced
        replaceable_indices = []
        for i, (word, pos) in enumerate(pos_tags):
            # Skip short words, punctuation, and numbers
            if len(word) <= 3 or not word.isalpha() or word.lower() in self.medical_stopwords:
                continue

            # Check if the word has a compatible POS tag
            if pos in self.pos_map:
                replaceable_indices.append(i)

        # Shuffle and limit the number of replacements
        random.shuffle(replaceable_indices)
        n_to_replace = min(n, len(replaceable_indices))
        indices_to_replace = replaceable_indices[:n_to_replace]

        # Replace words with synonyms
        new_words = words.copy()
        for i in indices_to_replace:
            word = words[i]
            pos = pos_tags[i][1]

            # Only replace with probability
            if random.random() > probability:
                continue

            # Try to get synonyms
            if pos in self.pos_map:
                wordnet_pos = self.pos_map[pos]
                synonyms = self._get_synonyms(word, wordnet_pos)
            else:
                synonyms = self._get_synonyms(word)

            # Replace the word if synonyms were found
            if synonyms:
                new_words[i] = random.choice(synonyms)

        # Reconstruct the text
        augmented_text = ' '.join(new_words)

        # Fix spacing around punctuation
        augmented_text = re.sub(r'\s+([.,;:!?)])', r'\1', augmented_text)
        augmented_text = re.sub(r'([(])\s+', r'\1', augmented_text)

        return augmented_text

    def augment_dataset(self, cases, num_augmentations=3):
        """
        Augment the dataset by creating multiple versions with synonym replacements
        Args:
            cases (list): List of case dictionaries
            num_augmentations (int): Number of augmented versions to create per case
        Returns:
            list: Original cases plus augmented cases
        """
        augmented_cases = []
        for case in tqdm(cases, desc="Augmenting dataset"):
            # Add the original case
            augmented_cases.append(case)

            # Create augmented versions
            for i in range(num_augmentations):
                aug_case = case.copy()

                # Create a new case ID for the augmented case
                aug_case['case_id'] = f"{case['case_id']}_aug{i + 1}"

                # Augment patient narrative
                if 'patient_narrative' in case:
                    aug_case['patient_narrative'] = self.synonym_replacement(
                        case['patient_narrative'],
                        n=max(3, len(case['patient_narrative'].split()) // 10),
                        probability=0.4
                    )

                # Augment clinician question
                if 'clinician_question' in case:
                    aug_case['clinician_question'] = self.synonym_replacement(
                        case['clinician_question'],
                        n=max(2, len(case['clinician_question'].split()) // 10),
                        probability=0.4
                    )

                # Augment sentences in note excerpt
                if 'sentences' in case:
                    aug_sentences = []
                    for sent in case['sentences']:
                        aug_sent = sent.copy()
                        aug_sent['text'] = self.synonym_replacement(
                            sent['text'],
                            n=max(2, len(sent['text'].split()) // 10),
                            probability=0.4
                        )
                        aug_sentences.append(aug_sent)
                    aug_case['sentences'] = aug_sentences

                augmented_cases.append(aug_case)

        return augmented_cases


# For backward compatibility
MedicalDataAugmenter = SimpleMedicalAugmenter