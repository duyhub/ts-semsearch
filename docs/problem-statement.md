Tasco

Mobility Track / Problem statement

# P7) AI Semantic Search & Ranking

Users search by needs and attributes, not only exact place names, so keyword search often misses intent.

## Platform Context

This challenge is built around Tasco Maps, Vietnam's next-generation digital map platform designed to help users discover places, businesses, services, and mobility experiences.

Tasco Maps aims to provide intelligent search, local discovery, recommendations, navigation, and location-based experiences tailored for Vietnamese users and businesses.

## Platform Resources

This challenge is built on top of the Tasco Maps ecosystem. Participants are encouraged to explore the Tasco Maps application and design solutions that enhance search, discovery, recommendations, AI experiences, and local content rather than replacing the underlying map platform.

Participants are encouraged to build solutions that are integration-ready with the Tasco Maps ecosystem.

## Objective

Build an AI-powered retrieval and ranking system that helps users discover the most relevant places, businesses, and services.

Users often do not know the exact name of the place they are looking for. Instead, they search using needs, preferences, attributes, or natural language descriptions.

The solution should understand the meaning behind queries and return the most relevant results rather than relying solely on keyword matching.

## Core capabilities

- Semantic Search: understand query meaning beyond keyword matching.
- Vector Search: retrieve results using vector embeddings.
- Embeddings: represent queries and POIs in semantic vector space.
- AI Ranking: rank search results using AI models.
- Relevance Optimization: improve search result relevance.
- Candidate Retrieval: retrieve relevant candidate results.
- Re-ranking: reorder results using ranking signals.

## Example user scenarios

- quiet coffee shop for work -> coffee shops suitable for working, with Wi-Fi and quiet environment.
- hotel near da nang beach -> hotels located near Da Nang beach.
- place for a date -> romantic restaurants, cafes, or rooftop venues.
- restaurants open late -> restaurants operating after 10 PM.
- places for children -> family-friendly attractions and entertainment venues.
- coffee shop with parking -> coffee shops with parking facilities.
- Ranking signals include relevance, distance, popularity, ratings, review insights, freshness, and business attributes.

## Expected output

- For each query, return ranked search results with relevance explanations.
- Example for quiet coffee shop for work should return ranked POIs with relevance score and reasons such as Wi-Fi available, quiet environment, and work-friendly.

## Expected deliverables

- Semantic Search Engine.
- Retrieval Engine.
- Ranking Engine.
- Search API / Service.
- Live demo of search quality and ranking.

## Submission requirements

- Presentation deck.
- Live demonstration or recorded video.
- Source code repository.
- README with solution overview, setup instructions, and technologies used.
- At least 10 sample search queries and ranked search results.
- Explanation of retrieval and ranking methodology.
- Description of ranking signals and AI models used.

## Suggested architecture

- Query Input: user search query.
- Embedding Layer: generate query embeddings.
- Retrieval Layer: retrieve candidate results.
- Vector Database: store embeddings.
- Ranking Layer: AI-based ranking.
- Search API: return ranked results.

## Success criteria

- Understand semantic meaning beyond keywords.
- Retrieve relevant places and businesses.
- Rank results effectively using multiple signals.
- Provide explainable search results.
- Return fast and relevant results.
- Support real-world search experiences.

## Provided resources

- Search Query Dataset covering different search intents.
- POI Dataset with places, businesses, categories, locations, brands, and attributes.
- POI Metadata with ratings, popularity signals, descriptions, and tags.

Build direction

Build semantic retrieval and ranking for places using embeddings, relevance signals, distance, ratings, attributes, and explanations.

Resources

## Downloads and reference links

### AI Maps Challenge Package

Dataset Download

View Resource, ↗

### App Store

Resource Links

View Resource, ↗

### API Documentation

Resource Links

View Resource, ↗

## Attached data

Downloaded from the linked Google Drive resource, kept in this problem folder:

- `data/ai_maps_track2_dataset_participants.xlsx`

Shared docs for this problem group:

- `../_shared/maps-api/tasco_maps_hackathon_api_documentation.pdf`

---

_Source: https://aitalent.genaifund.ai/tracks/mobility/maps-semantic-ranking_
