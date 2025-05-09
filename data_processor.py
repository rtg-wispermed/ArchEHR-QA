# data_processor.py
import xml.etree.ElementTree as ET
import json
import pandas as pd

class ArchEHRDataProcessor:
    def __init__(self, xml_path, key_path=None):
        self.xml_path = xml_path
        self.key_path = key_path

    def load_data(self):
        # Parse XML file
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing XML file {self.xml_path}: {e}")
            return [], {}
        except FileNotFoundError:
            print(f"Error: XML file not found at {self.xml_path}")
            return [], {}

        cases = []
        for case in root.findall('case'):
            case_id = case.get('id')

            # Extract and clean patient_narrative
            patient_narrative_tag = case.find('patient_narrative')
            patient_narrative_raw = patient_narrative_tag.text if patient_narrative_tag is not None else None
            patient_narrative = ' '.join(patient_narrative_raw.split()) if patient_narrative_raw else ""

            # Extract and clean clinician_question
            clinician_question_tag = case.find('clinician_question')
            clinician_question_raw = clinician_question_tag.text if clinician_question_tag is not None else None
            clinician_question = ' '.join(clinician_question_raw.split()) if clinician_question_raw else ""

            # Extract and clean note_excerpt (optional, depends if you use the full excerpt)
            note_excerpt_tag = case.find('note_excerpt')
            note_excerpt_raw = note_excerpt_tag.text if note_excerpt_tag is not None else None
            note_excerpt = ' '.join(note_excerpt_raw.split()) if note_excerpt_raw else ""

            # Extract and clean sentences
            sentences = []
            sentences_container = case.find('.//note_excerpt_sentences')
            if sentences_container is not None:
                for sentence in sentences_container.findall('sentence'):
                    sentence_id = sentence.get('id')
                    sentence_text_raw = sentence.text
                    # Normalize whitespace (removes line breaks, extra spaces)
                    sentence_text = ' '.join(sentence_text_raw.split()) if sentence_text_raw else ""
                    sentences.append({
                        'id': sentence_id,
                        'text': sentence_text
                    })

            cases.append({
                'case_id': case_id,
                'patient_narrative': patient_narrative,
                'clinician_question': clinician_question,
                # Store the cleaned full excerpt if needed, otherwise might not be necessary
                # 'note_excerpt': note_excerpt,
                'sentences': sentences
            })

        # Load relevance keys if available
        relevance_data = {}
        if self.key_path:
            try:
                with open(self.key_path, 'r') as f:
                    key_data = json.load(f)
                for item in key_data: # Changed variable name from 'case' to 'item'
                    case_id = item.get('case_id')
                    if case_id:
                        answers = {ans_item['sentence_id']: ans_item['relevance']
                                   for ans_item in item.get('answers', []) if 'sentence_id' in ans_item}
                        relevance_data[case_id] = answers
            except FileNotFoundError:
                print(f"Warning: Key file not found at {self.key_path}, proceeding without relevance data.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON key file {self.key_path}: {e}")
                # Decide whether to proceed without relevance or raise error
            except Exception as e:
                 print(f"An unexpected error occurred loading key file {self.key_path}: {e}")


        return cases, relevance_data

    def prepare_for_model(self, cases, relevance_data=None):
        """Convert data to format suitable for model training"""
        examples = []
        for case in cases:
            case_id = case.get('case_id')
            question = case.get('clinician_question', '') # Use .get for safety
            sentences = case.get('sentences', [])

            # Create features for each sentence
            for sentence in sentences:
                sentence_id = sentence.get('id')
                sentence_text = sentence.get('text', '')

                # Get relevance label if available
                relevance = "unknown"
                if relevance_data and case_id in relevance_data:
                    # Check if sentence_id exists for this case in relevance_data
                    if sentence_id in relevance_data[case_id]:
                        relevance = relevance_data[case_id][sentence_id]

                examples.append({
                    'case_id': case_id,
                    'question': question,
                    'sentence_id': sentence_id,
                    'sentence_text': sentence_text,
                    'relevance': relevance
                })

        return pd.DataFrame(examples)

