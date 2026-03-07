QA_generation_prompt = """
You are shown **one page image extracted from a PDF**.  
Write **exactly one fact-based question** and its answer for use in **Retrieval-Augmented Generation (RAG) evaluation**.

Requirements
1. **Grounded** – The answer MUST be found verbatim in the image (no reasoning or external knowledge).
2. **Specific** – Ask for a concrete item: a date, name, figure, heading, or short phrase.
3. **Search-style** – Word the question like something typed into a search engine.
4. **Standalone** – The question must be fully understandable without mentioning any “image”, “page”, “document”, or “context”.
5. **Concise answer** – Respond with the minimal text snippet (≤ 20 tokens) that appears in the image.

Return your result in **exactly** this format and nothing else:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the PDF image.

Output:::"""

question_groundedness_critique_prompt = """
You are shown **one page image extracted from a PDF** and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given PDF image.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the PDF image.

Return your result in **exactly** this format and nothing else:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

Now here are the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.

Return your result in **exactly** this format and nothing else:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """