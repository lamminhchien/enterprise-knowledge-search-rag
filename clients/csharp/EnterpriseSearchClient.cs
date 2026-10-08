using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace EnterpriseSearch.Client
{
    public class QueryRequest
    {
        [JsonPropertyName("query")]
        public string Query { get; set; } = string.Empty;

        [JsonPropertyName("top_k")]
        public int TopK { get; set; } = 3;

        [JsonPropertyName("user_role")]
        public string UserRole { get; set; } = "employee";
    }

    public class SearchResultItem
    {
        [JsonPropertyName("filename")]
        public string Filename { get; set; } = string.Empty;

        [JsonPropertyName("text")]
        public string Text { get; set; } = string.Empty;

        [JsonPropertyName("rrf_score")]
        public double RrfScore { get; set; }
    }

    public class SearchResponse
    {
        [JsonPropertyName("query")]
        public string Query { get; set; } = string.Empty;

        [JsonPropertyName("sanitized_query")]
        public string SanitizedQuery { get; set; } = string.Empty;

        [JsonPropertyName("results")]
        public List<SearchResultItem> Results { get; set; } = new();
    }

    /// <summary>
    /// Lightweight C# client integrating with the Enterprise Knowledge Search & RAG REST API.
    /// Demonstrates async HTTP communication and strong typing.
    /// </summary>
    public class EnterpriseSearchClient
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public EnterpriseSearchClient(string baseUrl = "http://localhost:8000")
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _httpClient = new HttpClient();
        }

        public async Task<SearchResponse?> ExecuteSearchAsync(string query, int topK = 3)
        {
            var requestPayload = new QueryRequest { Query = query, TopK = topK };
            var jsonContent = new StringContent(
                JsonSerializer.Serialize(requestPayload),
                Encoding.UTF8,
                "application/json"
            );

            var response = await _httpClient.PostAsync($"{_baseUrl}/api/v1/search", jsonContent);
            response.EnsureSuccessStatusCode();

            var responseString = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<SearchResponse>(responseString);
        }

        public static async Task Main(string[] args)
        {
            Console.WriteLine("=== AvePoint Enterprise Knowledge Search: C# Client Demo ===");
            var client = new EnterpriseSearchClient();

            try
            {
                var query = "Quy định bảo mật thông tin và MFA";
                Console.WriteLine($"Sending Query: '{query}'");

                var result = await client.ExecuteSearchAsync(query);
                if (result != null)
                {
                    Console.WriteLine($"Sanitized Query: {result.SanitizedQuery}");
                    Console.WriteLine($"Retrieved {result.Results.Count} evidence items:");
                    foreach (var item in result.Results)
                    {
                        Console.WriteLine($"  - Source: {item.Filename} (RRF Score: {item.RrfScore:F4})");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"API Request note: Ensure FastAPI is running on port 8000 ({ex.Message})");
            }
        }
    }
}
