from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
import re
import nltk
from nltk.data import load as nltk_load
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from rouge_score import rouge_scorer
from torch.nn.functional import cosine_similarity

# Ensure NLTK Punkt data is downloaded
try:
    nltk_load('tokenizers/punkt/english.pickle')
except LookupError:
    print("[INFO] NLTK 'punkt' tokenizer data not found. Downloading...")
    nltk.download('punkt', quiet=True)

class AnswerGenerator:
    def __init__(self,
                 generator_model_name="Mahalingam/DistilBart-Med-Summary",
                 embedding_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):

        # --- Generator Model ---
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Answer generator loaded on: {self.device}")

        # --- Embedding Model (for citations) ---
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embed_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.embed_model.eval()
        print(f"Citation embedding model loaded on: {self.device}")

        # --- ROUGE Scorer ---
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        print("ROUGE scorer initialized for citation matching.")

        # --- Initialize and Customize NLTK Sentence Tokenizer ---
        print("Initializing and customizing NLTK sentence tokenizer...")
        try:
            # Load the default English Punkt tokenizer parameters
            punkt_param = PunktParameters()

            # === ADD ABBREVIATIONS HERE ===
            # Add abbreviations WITHOUT the final period.
            abbreviations = ['e.g', 'i.e', 'dr', 'mr', 'mrs', 'ms', 'prof', 'inc',
                             'corp', 'vs', 'etc', 'al', 'p.o', 't.i.d'] # Keep existing custom abbreviations
            # Add specific terms if needed, e.g. 'fig', 'tab' if they cause issues.
            punkt_param.abbrev_types.update(abbreviations)
            # =============================

            # Create a tokenizer instance with the customized parameters
            self.custom_sentence_tokenizer = PunktSentenceTokenizer(punkt_param)
            print(f"Custom NLTK sentence tokenizer initialized with {len(abbreviations)} custom abbreviations.")

        except Exception as e:
            print(f"[ERROR] Failed to initialize custom NLTK sentence tokenizer: {e}. Will fallback to default NLTK.")
            self.custom_sentence_tokenizer = None # Fallback indicator
        # ---------------------------------------------------------

    def generate_answer(self, question, relevant_sentences, sentence_ids, case_id="unknown"):
        # Prepare context
        context = " ".join(relevant_sentences)

        # Flan-T5 Prompt
        #prompt = f"""Question: {question}

        #Context: {context}
        #Instructions:
        #1. Create a comprehensive, narrative answer in paragraph form to the question based STRICTLY on the provided context sentences.
        #2. Use complete sentences. Do NOT use lists.
        #3. Every sentence in your answer MUST be directly supported by evidence from the context.
        #4. Minimize paraphrasing. Prefer using exact phrases from the context for medical terms, findings, and actions.
        #5. The answer must not exceed 75 words.
        #6. Preserve all medical terminology exactly as it appears. Do not simplify.
        #7. Ensure clinical accuracy and a professional tone.

        #Answer:
        #"""

        # BART Prompt
        prompt = (f"{context} Based on the text above, answer the question: {question}\n"
                  f"Answer:")

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True,
            repetition_penalty=1.2, no_repeat_ngram_size=3,
        )

        raw_decoded_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        sentences_raw = re.split(r'(?<=[.!?])\s*', raw_decoded_answer.strip())
        filtered_sentences = []
        prefix_to_remove = "Based on the text above"
        for sent in sentences_raw:
            if not sent or not sent.strip(): continue
            sent_strip = sent.strip()
            if sent_strip.startswith(prefix_to_remove):
                print(f"[DEBUG] Removing sentence starting with prefix (Case {case_id}): '{sent_strip}'")
                continue
            if '?' in sent_strip:
                print(f"[DEBUG] Removing sentence containing '?' (Case {case_id}): '{sent_strip}'")
                continue
            filtered_sentences.append(sent_strip)
        answer = " ".join(filtered_sentences)

        answer = answer.replace("\u00a0", " ")
        answer = re.sub(r'\s+', ' ', answer).strip()

        print(f"[DEBUG] Answer after filtering & normalization (Case {case_id}):\n'{answer}'")

        # Call add_citations with the cleaned answer
        answer_with_citations = self.add_citations(answer, relevant_sentences, sentence_ids, top_n=3)
        return answer_with_citations


    # === ADD_CITATIONS METHOD WITH SENTENCE MERGING ===
    def add_citations(self, answer, relevant_sentences, sentence_ids, top_n=3):
        if not answer or not answer.strip():
            return ""

        # Step 1: Tokenize using the customized NLTK tokenizer
        try:
            if self.custom_sentence_tokenizer:
                initial_sentences = self.custom_sentence_tokenizer.tokenize(answer)
            else:
                print("[WARNING] Using default NLTK sentence tokenizer due to initialization failure.")
                initial_sentences = nltk.sent_tokenize(answer)
        except Exception as e:
            print(f"[ERROR] Sentence tokenization failed: {e}. Falling back to simple regex split.")
            # Fallback regex preserves terminators better for merging check
            initial_sentences = re.split(r'([.!?])\s*', answer.strip())
            # Reconstruct sentences from split parts if regex fallback is used
            if len(initial_sentences) > 1:
                 temp_sentences = []
                 for i in range(0, len(initial_sentences) - 1, 2):
                      sent_part = initial_sentences[i]
                      term_part = initial_sentences[i+1] if i+1 < len(initial_sentences) else ''
                      temp_sentences.append((sent_part + term_part).strip())
                 # Add last part if length is odd
                 if len(initial_sentences) % 2 != 0 and initial_sentences[-1].strip():
                     temp_sentences.append(initial_sentences[-1].strip())
                 initial_sentences = [s for s in temp_sentences if s]


        # === Step 2: Merge fragmented sentences ===
        merged_sentences = []
        i = 0
        while i < len(initial_sentences):
            current_sent = initial_sentences[i].strip()
            if not current_sent:
                i += 1
                continue

            # Check if the *next* sentence exists and is potentially a fragment
            if i + 1 < len(initial_sentences):
                next_sent = initial_sentences[i+1].strip()
                # Define "short" (e.g., <= 5 words, slightly more lenient)
                is_short_fragment = len(next_sent.split()) <= 5
                # Pattern: ends with a number followed by a period. More general than just pH.
                ends_with_numeric = re.search(r'\b\d+(\.\d+)?\.$', current_sent)
                # Optional: Add a check that the fragment doesn't start like a new sentence (e.g., Capital letter)
                # starts_lowercase_or_number = next_sent and (not next_sent[0].isupper() or next_sent[0].isdigit())

                # Merge if current ends with numeric pattern and next is short
                if ends_with_numeric and is_short_fragment: # and starts_lowercase_or_number:
                    print(f"[DEBUG] Merging fragmented sentences: '{current_sent}' + '{next_sent}'")
                    merged_sentences.append(current_sent + " " + next_sent)
                    i += 2 # Skip the next sentence as it's merged
                    continue

            # If no merge condition met, add the current sentence as is
            merged_sentences.append(current_sent)
            i += 1
        # =========================================

        # Step 3: Process merged sentences for citations
        answer_with_citations = []
        print(f"[DEBUG] Sentences after merging (for citation): {merged_sentences}") # Debug print

        # Use merged_sentences instead of initial_sentences for the loop
        for ans_sent_strip in merged_sentences:
            if not ans_sent_strip:
                continue

            # Find matches using ROUGE score
            matches_with_scores = self.find_matches(ans_sent_strip, relevant_sentences)

            if matches_with_scores:
                top_matches = matches_with_scores[:top_n]
                citations_ids_int = sorted([int(sentence_ids[idx]) for idx, score in top_matches])
                citations = " |" + ",".join(map(str, citations_ids_int)) + "|"
                ans_sent_with_citation = ans_sent_strip + citations
            else:
                ans_sent_with_citation = ans_sent_strip

            answer_with_citations.append(ans_sent_with_citation)

        return "\n".join(answer_with_citations)
    # ========================================================

    def _get_embedding(self, text):
        if not text or not text.strip():
             return torch.zeros(self.embed_model.config.hidden_size).to(self.device)
        inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
             outputs = self.embed_model(**inputs)
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled.squeeze()

    def find_matches(self, answer_sentence, context_sentences, threshold=0.3):
        if not answer_sentence or not answer_sentence.strip():
            return []
        matches_with_scores = []
        for i, context_sent in enumerate(context_sentences):
            if not context_sent or not context_sent.strip():
                continue
            rouge_scores = self.rouge_scorer.score(context_sent, answer_sentence)
            score = rouge_scores['rougeL'].fmeasure
            if score > threshold:
                matches_with_scores.append((i, score))
        # Sort matches by ROUGE score *after* checking all context sentences
        matches_with_scores.sort(key=lambda item: item[1], reverse=True)
        return matches_with_scores

