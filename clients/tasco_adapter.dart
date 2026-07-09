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
      // Contract-exact ErrorResponse: { error: {code, message}, requestId }
      final err = jsonDecode(res.body) as Map<String, dynamic>;
      throw TascoApiException(res.statusCode, err['error']?['code'], err['error']?['message']);
    }
    final body = jsonDecode(res.body) as Map<String, dynamic>;
    final results = (body['results'] as List).cast<Map<String, dynamic>>();
    return results.map(SearchSuggestion.fromPlaceResult).toList();
  }
}

class TascoApiException implements Exception {
  final int status;
  final String? code;
  final String? message;
  TascoApiException(this.status, this.code, this.message);
  @override
  String toString() => 'TascoApiException($status $code): $message';
}
