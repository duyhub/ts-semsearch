// Tasco Maps client adapter (PRD FR-16; tasco_api.pdf mapping table).
//
// Drop-in for the Flutter app's search layer: point `baseUrl` at this engine's
// /v1/search and map the contract-exact PlaceResult -> the app's SearchSuggestion.
// Integration is a base-URL change; no UI dependencies.
//
//   final client = TascoSemanticClient(baseUrl: 'https://semsearch.example.com');
//   final suggestions = await client.search('quán cà phê yên tĩnh để làm việc',
//                                            lat: 10.7738, lon: 106.7040);
//
// PlaceResult -> SearchSuggestion mapping (per the PDF):
//   id          -> id
//   name/label  -> label
//   category    -> meta.category ; type -> meta.type
//   address     -> description
//   coordinates -> coordinates (WGS84, unchanged)

import 'dart:convert';
import 'package:http/http.dart' as http;

class Coordinates {
  final double lat;
  final double lon;
  const Coordinates(this.lat, this.lon);
}

/// Mirrors the app's existing SearchSuggestion DTO.
class SearchSuggestion {
  final String id;
  final String label;
  final String description;
  final Coordinates coordinates;
  final Map<String, dynamic> meta;

  const SearchSuggestion({
    required this.id,
    required this.label,
    required this.description,
    required this.coordinates,
    required this.meta,
  });

  /// Map a contract-exact PlaceResult JSON object to a SearchSuggestion.
  factory SearchSuggestion.fromPlaceResult(Map<String, dynamic> p) {
    final c = p['coordinates'] as Map<String, dynamic>;
    return SearchSuggestion(
      id: p['id'] as String, // stable, e.g. "poi:C001"
      label: (p['label'] ?? p['name']) as String, // diacritics preserved
      description: p['address'] as String,
      coordinates: Coordinates(
        (c['lat'] as num).toDouble(),
        (c['lon'] as num).toDouble(),
      ),
      meta: {
        'type': p['type'],
        'category': p['category'],
        'score': p['score'],
        'distanceMeters': p['distanceMeters'],
        'source': p['source'],
        'tags': p['tags'],
      },
    );
  }
}

class TascoSemanticClient {
  final String baseUrl;
  final String? apiKey; // optional; sent as X-API-Key / Bearer when configured
  final http.Client _http;

  TascoSemanticClient({required this.baseUrl, this.apiKey, http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  Future<List<SearchSuggestion>> search(
    String query, {
    double? lat,
    double? lon,
    int limit = 10,
    String lang = 'vi',
  }) async {
    if ((lat == null) != (lon == null)) {
      throw ArgumentError('lat and lon must be supplied together');
    }
    final params = <String, String>{
      'q': query,
      'limit': '$limit',
      'lang': lang,
      if (lat != null) 'lat': '$lat',
      if (lon != null) 'lon': '$lon',
    };
    final uri = Uri.parse('$baseUrl/v1/search').replace(queryParameters: params);
    final headers = <String, String>{
      'X-Request-Id': DateTime.now().microsecondsSinceEpoch.toString(),
      if (apiKey != null) 'Authorization': 'Bearer $apiKey',
    };

    final res = await _http.get(uri, headers: headers);
    if (res.statusCode != 200) {
      throw _errorFor(res);
    }
    final body = jsonDecode(res.body) as Map<String, dynamic>;
    final results = (body['results'] as List).cast<Map<String, dynamic>>();
    return results.map(SearchSuggestion.fromPlaceResult).toList();
  }

  /// Build a structured exception from a non-200 response. The body is usually the
  /// contract ErrorResponse `{ error: {code, message}, requestId }`, but a proxy or
  /// gateway can return HTML/plain text — so JSON parsing is guarded and falls back
  /// to a body snippet. The response's X-Request-Id is surfaced when present.
  TascoApiException _errorFor(http.Response res) {
    final requestId = res.headers['x-request-id']; // http lowercases header keys
    try {
      final err = jsonDecode(res.body) as Map<String, dynamic>;
      final e = err['error'];
      return TascoApiException(
        res.statusCode,
        (e is Map) ? e['code'] as String? : null,
        (e is Map) ? e['message'] as String? : null,
        requestId: requestId ?? (err['requestId'] as String?),
      );
    } catch (_) {
      // Non-JSON body (e.g. an upstream proxy's HTML error page).
      final b = res.body;
      final snippet = b.length > 200 ? '${b.substring(0, 200)}...' : b;
      return TascoApiException(res.statusCode, null, snippet, requestId: requestId);
    }
  }
}

class TascoApiException implements Exception {
  final int status;
  final String? code;
  final String? message;
  final String? requestId;
  TascoApiException(this.status, this.code, this.message, {this.requestId});
  @override
  String toString() {
    final req = requestId != null ? ' req=$requestId' : '';
    return 'TascoApiException($status ${code ?? '-'}$req): $message';
  }
}
