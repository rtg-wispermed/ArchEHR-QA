import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


class RAGPipeline:
    def __init__(self,
                 retriever_model_name="pritamdeka/S-PubMedBert-MS-MARCO",
                 generator_model_name="google/flan-t5-large"):
        # Initialize retriever model for semantic search
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
        self.retriever_model = AutoModel.from_pretrained(retriever_model_name)

        # Initialize generator model for answer generation
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever_model = self.retriever_model.to(self.device)
        self.generator_model = self.generator_model.to(self.device)

        print(f"RAG models loaded on: {self.device}")

    def _get_embeddings(self, texts):
        """Get embeddings for a list of texts using the retriever model"""
        inputs = self.retriever_tokenizer(texts, padding=True, truncation=True,
                                          return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.retriever_model(**inputs)
            # Use CLS token embedding as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

    def retrieve_relevant_sentences(self, question, sentences, top_k=5):
        """Retrieve the most relevant sentences for a question"""
        # Get embeddings for question and sentences
        question_embedding = self._get_embeddings([question])
        sentence_embeddings = self._get_embeddings(sentences)

        # Calculate similarity scores
        similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]

        # Get indices of top-k most similar sentences
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return relevant sentences and their indices
        relevant_sentences = [sentences[i] for i in top_indices]
        relevance_scores = [similarities[i] for i in top_indices]

        return relevant_sentences, top_indices.tolist(), relevance_scores

    def generate_answer(self, question, relevant_sentences, sentence_ids):
        """Generate an answer based on the question and relevant sentences"""
        # Combine relevant sentences into context
        context = " ".join(relevant_sentences)

        # Create prompt for the generator
        prompt = f"""Question: {question}
        Context: {context}
        Create a comprehensive, multi-sentence answer to the question based on ALL the provided context. 
        Each sentence in your answer should always be supported by evidence from the context. 
        Don't give unsupported sentences. 
        Supplementary information should also be included in the answer.
        Your answer should be professional and thorough."""

        # Generate answer
        inputs = self.generator_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.generator_model.generate(
                inputs["input_ids"],
                max_length=512,
                min_length=50,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )

        answer = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Add citations to the answer
        answer_with_citations = self.add_citations(answer, relevant_sentences, sentence_ids)

        return answer_with_citations

    def find_matches(self, answer_sentence, context_sentences, threshold=0.7):
        """Find context sentences that truly match the answer sentence content"""
        # Get embeddings for answer sentence and context sentences
        answer_embedding = self._get_embeddings([answer_sentence])
        context_embeddings = self._get_embeddings(context_sentences)

        # Calculate semantic similarity
        similarities = cosine_similarity(answer_embedding, context_embeddings)[0]

        # Find potential matches based on semantic similarity
        potential_matches = [(i, score) for i, score in enumerate(similarities) if score > threshold]

        # Secondary verification with lexical overlap
        verified_matches = []
        answer_words = set(re.findall(r'\b\w+\b', answer_sentence.lower()))

        for idx, score in potential_matches:
            context_sent = context_sentences[idx]
            context_words = set(re.findall(r'\b\w+\b', context_sent.lower()))

            # Calculate Jaccard similarity for lexical verification
            if len(answer_words) > 0 and len(context_words) > 0:
                overlap = len(answer_words.intersection(context_words))
                # Require at least 3 words in common or 20% overlap
                if overlap >= 6 or (overlap / len(answer_words) > 0.4):
                    verified_matches.append(idx)

        return verified_matches

    def add_citations(self, answer, relevant_sentences, sentence_ids):
        """Add citations to the answer based on evidence from relevant sentences"""
        # Split answer into sentences
        answer_sentences = re.split(r'(?<=[.!?])\s+', answer)

        # For each sentence in the answer, find the most relevant sentence from context
        answer_with_citations = []
        for ans_sent in answer_sentences:
            if not ans_sent.strip():
                continue

            # Find all matching sentences in the context
            matches = self.find_matches(ans_sent, relevant_sentences)

            if matches:
                # Add all relevant citations
                citations = " |" + "|".join([sentence_ids[idx] for idx in matches]) + "|"
                ans_sent_with_citation = ans_sent + citations
            else:
                ans_sent_with_citation = ans_sent

            answer_with_citations.append(ans_sent_with_citation)

        return "\n".join(answer_with_citations)
