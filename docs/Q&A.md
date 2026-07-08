# Questions
My team is currently studying the AI Semantic Search & Ranking track and has a few questions. We'd really appreciate your anwers:
1. Evaluation criteria and methodology: What specific metrics will the Organizing Committee use to evaluate the accuracy of the search/ranking system (e.g., Precision@K, NDCG, MRR, Recall)? Please provide the exact formulas and scoring methodology so teams can self-evaluate and optimize before submission.
2. Nature of the POI dataset: Is the provided POI dataset (places, businesses) real-world data (sourced from Tasco Maps or actual production data), or is it synthetic/mock data intended only for demo purposes?
3. Scope of allowed data usage: Are teams required to use only the data in the ai_maps_track2_dataset_participants file provided by the Organizing Committee? Is there an API from the Organizing Committee that teams are allowed to call? And are teams permitted to supplement or enrich the dataset with external data sources (e.g., public data, self-collected data)?
4. Expected output format: Could you clarify the exact output structure expected for each query — including the format (JSON, table, etc.) and the complete list of required fields (e.g., POI ID, name, relevance score, relevance explanation/reasons, distance, rating, etc.)?

# Answers
1. Evaluation methodology

This is a hackathon, not a research benchmark, so teams are not required to optimize for specific metrics such as Precision@K, NDCG or MRR.

Please use the Public Evaluation Dataset (expected focus/task type) to validate your solution. During judging, we will also use Hidden Evaluation Scenarios.

Evaluation will focus on:

Search relevance & semantic understanding
Retrieval & ranking quality
Explainable ranking
User experience
Technical design & production readiness

Teams may include their own evaluation metrics, but this is optional.

2. POI Dataset

The provided POI dataset is a synthetic dataset created specifically for the hackathon to simulate real-world scenarios.

3. Data Usage

Teams should use the provided dataset as the primary dataset. Public data may be used to enrich or normalize it. No official Search API is provided, teams are free to build their own retrieval and ranking pipeline.

4. Expected Output

The JSON in the challenge document is a recommended example, not a mandatory schema. Since the challenge targets production-ready Search APIs and Maps SDK integration, we recommend exposing results through a JSON API, but teams may extend the schema or present results differently. The exact format will not be evaluated; what matters is that the solution clearly demonstrates semantic understanding, retrieval, ranking, and explainable results.