import random
import nltk
from nltk.corpus import wordnet
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch



class MedicalEntityRecognizer:
    def __init__(self, model_name="d4data/biomedical-ner-all"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"Medical Entity Recognizer loaded on: {self.device}")

    def identify_entities(self, text):
        """Identify medical entities in the text using the pre-trained model"""
        if not text or len(text) < 10:
            return []

        try:
            entities = self.ner_pipeline(text)
            # Extract entity text and type
            extracted_entities = []
            for entity in entities:
                extracted_entities.append({
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'score': entity['score']
                })
            return extracted_entities
        except Exception as e:
            print(f"Error in entity recognition: {e}")
            return []


class EnhancedMedicalAugmenter:
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

        # Initialize the medical entity recognizer
        self.entity_recognizer = MedicalEntityRecognizer()

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

    def medical_concept_preserving_augmentation(self, text, probability=0.3):
        """Augment text while preserving medical concepts"""
        if not text or len(text) < 10:
            return text

        # Identify medical entities
        medical_entities = self.entity_recognizer.identify_entities(text)

        # Create a list of spans to protect from replacement
        protected_spans = []
        for entity in medical_entities:
            # Find all occurrences of the entity text in the original text
            for match in re.finditer(re.escape(entity['text']), text):
                protected_spans.append((match.start(), match.end()))

        # Sort spans by start position
        protected_spans.sort()

        # Tokenize the text
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)

        # Map token positions to character positions
        token_char_map = []
        pos = 0
        for word in words:
            # Find the word in the text starting from the current position
            while pos < len(text) and text[pos:pos + len(word)] != word:
                pos += 1
            if pos < len(text):
                token_char_map.append((pos, pos + len(word)))
                pos += len(word)
            else:
                # If we can't find the exact position, use an approximate one
                token_char_map.append((-1, -1))

        # Identify tokens that can be replaced (not in protected spans)
        replaceable_indices = []
        for i, ((start, end), (word, pos)) in enumerate(zip(token_char_map, pos_tags)):
            # Skip tokens with unknown positions
            if start == -1:
                continue

            # Check if the token is in a protected span
            is_protected = False
            for p_start, p_end in protected_spans:
                if (start >= p_start and start < p_end) or (end > p_start and end <= p_end):
                    is_protected = True
                    break

            # Skip protected tokens, short words, punctuation, and medical stopwords
            if is_protected or len(word) <= 3 or not word.isalpha() or word.lower() in self.medical_stopwords:
                continue

            # Check if the word has a compatible POS tag
            if pos in self.pos_map:
                replaceable_indices.append(i)

        # Shuffle and limit the number of replacements
        random.shuffle(replaceable_indices)
        n_to_replace = max(1, int(len(replaceable_indices) * probability))
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
        Augment the dataset by creating multiple versions with medical entity preserving augmentations
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
                    aug_case['patient_narrative'] = self.medical_concept_preserving_augmentation(
                        case['patient_narrative'],
                        probability=0.4
                    )

                # Augment clinician question
                if 'clinician_question' in case:
                    aug_case['clinician_question'] = self.medical_concept_preserving_augmentation(
                        case['clinician_question'],
                        probability=0.4
                    )

                # Augment sentences in note excerpt
                if 'sentences' in case:
                    aug_sentences = []
                    for sent in case['sentences']:
                        aug_sent = sent.copy()
                        aug_sent['text'] = self.medical_concept_preserving_augmentation(
                            sent['text'],
                            probability=0.4
                        )
                        aug_sentences.append(aug_sent)
                    aug_case['sentences'] = aug_sentences

                augmented_cases.append(aug_case)

        return augmented_cases
