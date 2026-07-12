// Tasco Maps client adapter (PRD FR-16; tasco_api.pdf mapping table).
//
// Drop-in for a Node/TypeScript service's search layer: point baseUrl at this engine's
// /v1/search and map the contract-exact PlaceResult -> the app's SearchSuggestion.
// Integration is a base-URL change; no UI dependencies. Uses the global fetch (Node 18+).
//
//   import { TascoSemanticClient } from './tasco_adapter.mjs';
//   const client = new TascoSemanticClient({ baseUrl: 'https://semsearch.example.com' });
//   const suggestions = await client.search('quán cà phê yên tĩnh để làm việc',
//                                            { lat: 10.7738, lon: 106.7040 });
//
// PlaceResult -> SearchSuggestion mapping (per the PDF):
//   id          -> id
//   name/label  -> label
//   category    -> meta.category ; type -> meta.type
//   address     -> description
//   coordinates -> coordinates (WGS84, unchanged)

/** Map a contract-exact PlaceResult object to a SearchSuggestion. */
export function suggestionFromPlaceResult(p) {
  const c = p['coordinates'];
  return {
    id: p['id'], // stable, e.g. "poi:C001"
    label: p['label'] || p['name'], // diacritics preserved
    description: p['address'],
    coordinates: { lat: c['lat'], lon: c['lon'] }, // WGS84, unchanged
    meta: {
      type: p['type'],
      category: p['category'],
      score: p['score'],
      distanceMeters: p['distanceMeters'],
      source: p['source'],
      tags: p['tags'],
    },
  };
}

/** Structured non-200 from the engine (contract ErrorResponse, or a body snippet). */
export class TascoApiError extends Error {
  constructor(status, code, message, requestId) {
    const req = requestId ? ` req=${requestId}` : '';
    super(`TascoApiError(${status} ${code ?? '-'}${req}): ${message ?? ''}`);
    this.name = 'TascoApiError';
    this.status = status;
    this.code = code ?? null;
    this.requestId = requestId ?? null;
  }
}

export class TascoSemanticClient {
  // fetchImpl is injectable so tests can drive the client without a live socket.
  constructor({ baseUrl, apiKey = null, fetchImpl = fetch } = {}) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
    this._fetch = fetchImpl;
  }

  async search(query, { lat = null, lon = null, limit = 10, lang = 'vi' } = {}) {
    const params = new URLSearchParams({ q: query, limit: String(limit), lang });
    if (lat != null) params.set('lat', String(lat));
    if (lon != null) params.set('lon', String(lon));

    const url = `${this.baseUrl}/v1/search?${params}`;
    const headers = { 'X-Request-Id': String(Date.now()) };
    if (this.apiKey != null) headers['Authorization'] = `Bearer ${this.apiKey}`;

    const res = await this._fetch(url, { headers });
    if (!res.ok) throw await errorFor(res);

    const body = await res.json();
    return (body.results ?? []).map(suggestionFromPlaceResult);
  }
}

async function errorFor(res) {
  // http header keys are case-insensitive; fetch lowercases them.
  const requestId = res.headers.get('x-request-id');
  const text = await res.text();
  try {
    const err = JSON.parse(text);
    const e = err.error;
    return new TascoApiError(
      res.status,
      e?.code ?? null,
      e?.message ?? null,
      requestId ?? err.requestId ?? null,
    );
  } catch {
    // Non-JSON body (e.g. an upstream proxy's HTML error page).
    const snippet = text.length > 200 ? `${text.slice(0, 200)}...` : text;
    return new TascoApiError(res.status, null, snippet, requestId);
  }
}
