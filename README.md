# RAG Evaluation Framework

A RAG  agent over PDF documents, plus a framework to measure
how good it is: answer quality, retrieval accuracy, and hallucination.

Built with **LangChain** and **DeepEval**. Works with **OpenAI, Groq, Gemini or Ollama**,
including a fully free setup (Groq + Gemini).

## How it works

```
question ──► agent ──► search tool ──► vector store (PDF chunks)
                │
                ▼
             answer ──► LLM judge (DeepEval) ──► scores ──► experiments.json
```

1. The PDF is split into paragraphs and embedded into a vector store.
2. A tool-calling agent searches the store and answers each question from the dataset.
3. An LLM judge scores every answer. Retrieval is also scored without any LLM.
4. Each experiment is saved, so different settings can be compared.

## Metrics

| Metric | What it checks | 
|---|---|
| Answer Relevancy | Does the answer address the question? | 
| Faithfulness | Does the answer contradict the retrieved context? | 
| Contextual Relevancy | Is the retrieved context relevant to the question? | 
| Correctness | Does the answer match the expected answer? |
| Hit Rate @k | Is the chunk containing the answer among the top k results? | 
| MRR @k | How high is that chunk ranked? |

### Why a custom hallucination metric?

While testing, I gave the evaluation an invented answer: *"Emperor penguins grow to about
115 cm tall"* (the document never mentions height). **Faithfulness scored it 1.0.**

Faithfulness only penalizes claims that **contradict** the context. A claim the context
doesn't mention at all counts as fine, so invented facts get through.

## Results

**Retrieval** (24 questions, no LLM):

| Embeddings | Hit@1 | Hit@3 | MRR@3 |
|---|---|---|---|
| Gemini (`gemini-embedding-001`) | 1.00 | 1.00 | 1.00 |
| OpenAI (`text-embedding-3-large`) | 0.96 | 1.00 | 0.98 |
| Ollama (`nomic-embed-text`) | 0.92 | 0.96 | 0.94 |

**Agent with GPT-4o**, different retriever `k` (20 normal questions):

| Metric | k=1 | k=2 | k=3 |
|---|---|---|---|
| Answer Relevancy | 1.00 | 0.95 | 1.00 |
| Faithfulness | 1.00 | 1.00 | 1.00 |
| Contextual Relevancy | 0.77 | 0.78 | 0.75 |
| Correctness | 0.95 | 0.98 | 0.98 |

The current dataset is small (one PDF, 10 chunks), so most scores are close to 1.0.
Adding more documents to make retrieval harder is the next step.


Main settings in `.env`:

```bash
PROVIDER=groq                       # LLM for agent + judge: openai | groq | gemini | ollama
EMBEDDING_PROVIDER=gemini           # embeddings: openai | gemini | ollama
GROQ_MODEL=openai/gpt-oss-120b      # agent model
GROQ_JUDGE_MODEL=qwen/qwen3.8-27b   # optional: separate judge model (its own daily limit)
```

Free setup: `PROVIDER=groq` + `EMBEDDING_PROVIDER=gemini` (free API keys from Groq and Google AI Studio).


## Project structure

```
RAG1/
  agent.py           RAG agent (LangChain, tool calling)
  tools.py           search tool used by the agent
  vectorstore.py     PDF -> chunks -> embeddings
  indexing.py        vector store used by the agent
  judge.py           LLM judge for DeepEval (any LangChain model)
  prompt.py          system prompt
run_experiment.py    full experiment: answers -> evaluation -> summary
compare_experiments.py
eval_retrieval.py    retrieval metrics without an LLM
eval_dataset.jsonl   questions, expected answers, trick questions, evidence
experiments/         saved results (one folder per experiment)
single_tests/        examples testing one metric at a time
pdfs/                source documents
```

