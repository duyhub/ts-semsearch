// Package tasco is a Tasco Maps client adapter (PRD FR-16; tasco_api.pdf mapping table).
//
// Drop-in for a Go service's search layer: point BaseURL at this engine's /v1/search
// and map the contract-exact PlaceResult -> the app's SearchSuggestion. Integration is
// a base-URL change; no UI dependencies.
//
//	c := tasco.NewClient("https://semsearch.example.com")
//	lat, lon := 10.7738, 106.7040
//	sugg, err := c.Search(ctx, "quán cà phê yên tĩnh để làm việc",
//	    tasco.SearchOptions{Lat: &lat, Lon: &lon})
//
// PlaceResult -> SearchSuggestion mapping (per the PDF):
//
//	id          -> ID
//	name/label  -> Label
//	address     -> Description
//	category    -> Meta["category"] ; type -> Meta["type"]
//	coordinates -> Coordinates (WGS84, unchanged)
package tasco

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

type Coordinates struct {
	Lat float64 `json:"lat"`
	Lon float64 `json:"lon"`
}

// PlaceResult mirrors the contract-exact DTO from tasco_api.pdf.
type PlaceResult struct {
	ID             string      `json:"id"`
	Type           string      `json:"type"`
	Name           string      `json:"name"`
	Label          string      `json:"label"`
	Address        string      `json:"address"`
	Category       string      `json:"category"`
	Coordinates    Coordinates `json:"coordinates"`
	DistanceMeters *int        `json:"distanceMeters"`
	Score          float64     `json:"score"`
	Source         string      `json:"source"`
	Tags           []string    `json:"tags"`
}

// SearchSuggestion mirrors the app's existing DTO.
type SearchSuggestion struct {
	ID          string
	Label       string
	Description string
	Coordinates Coordinates
	Meta        map[string]any
}

func suggestionFromPlaceResult(p PlaceResult) SearchSuggestion {
	label := p.Label // diacritics preserved
	if label == "" {
		label = p.Name
	}
	return SearchSuggestion{
		ID:          p.ID, // stable, e.g. "poi:C001"
		Label:       label,
		Description: p.Address,
		Coordinates: p.Coordinates, // WGS84, unchanged
		Meta: map[string]any{
			"type":           p.Type,
			"category":       p.Category,
			"score":          p.Score,
			"distanceMeters": p.DistanceMeters,
			"source":         p.Source,
			"tags":           p.Tags,
		},
	}
}

// Client talks to this engine's /v1/search endpoint.
type Client struct {
	BaseURL string
	APIKey  string // optional; sent as Bearer when set
	HTTP    *http.Client
}

func NewClient(baseURL string) *Client {
	return &Client{BaseURL: baseURL, HTTP: &http.Client{Timeout: 5 * time.Second}}
}

// SearchOptions carries the optional query parameters. Lat/Lon are pointers so an
// unset location is omitted from the request (mirrors the Dart adapter).
type SearchOptions struct {
	Lat   *float64
	Lon   *float64
	Limit int    // defaults to 10
	Lang  string // defaults to "vi"
}

func (c *Client) Search(ctx context.Context, query string, opts SearchOptions) ([]SearchSuggestion, error) {
	if opts.Limit == 0 {
		opts.Limit = 10
	}
	if opts.Lang == "" {
		opts.Lang = "vi"
	}
	params := url.Values{}
	params.Set("q", query)
	params.Set("limit", strconv.Itoa(opts.Limit))
	params.Set("lang", opts.Lang)
	if opts.Lat != nil {
		params.Set("lat", strconv.FormatFloat(*opts.Lat, 'f', -1, 64))
	}
	if opts.Lon != nil {
		params.Set("lon", strconv.FormatFloat(*opts.Lon, 'f', -1, 64))
	}

	endpoint := c.BaseURL + "/v1/search?" + params.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Request-Id", strconv.FormatInt(time.Now().UnixMicro(), 10))
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	res, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, errorFor(res.StatusCode, res.Header.Get("X-Request-Id"), body)
	}

	var payload struct {
		Results []PlaceResult `json:"results"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("tasco: decoding /v1/search response: %w", err)
	}
	out := make([]SearchSuggestion, 0, len(payload.Results))
	for _, p := range payload.Results {
		out = append(out, suggestionFromPlaceResult(p))
	}
	return out, nil
}

// APIError is a structured non-200 from the engine. The body is usually the contract
// ErrorResponse { error: {code, message}, requestId }, but a proxy/gateway can return
// HTML/plain text — so JSON parsing is guarded and falls back to a body snippet.
type APIError struct {
	Status    int
	Code      string
	Message   string
	RequestID string
}

func (e *APIError) Error() string {
	req := ""
	if e.RequestID != "" {
		req = " req=" + e.RequestID
	}
	code := e.Code
	if code == "" {
		code = "-"
	}
	return fmt.Sprintf("tasco: API error (%d %s%s): %s", e.Status, code, req, e.Message)
}

func errorFor(status int, headerRequestID string, body []byte) *APIError {
	e := &APIError{Status: status, RequestID: headerRequestID}
	var parsed struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
		RequestID string `json:"requestId"`
	}
	if err := json.Unmarshal(body, &parsed); err == nil {
		e.Code = parsed.Error.Code
		e.Message = parsed.Error.Message
		if e.RequestID == "" {
			e.RequestID = parsed.RequestID
		}
		return e
	}
	// Non-JSON body (e.g. an upstream proxy's HTML error page).
	snippet := string(body)
	if len(snippet) > 200 {
		snippet = snippet[:200] + "..."
	}
	e.Message = snippet
	return e
}
