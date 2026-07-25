using System.Net.Http.Json;
using System.Text.Json.Serialization;
using MfluxExperiment.WebApi.Models;

namespace MfluxExperiment.WebApi.Services;

/// <summary>
/// HTTP client that talks to the local mflux FastAPI inference service.
/// </summary>
public class MfluxInferenceClient : IInferenceClient
{
    private readonly HttpClient _http;

    public MfluxInferenceClient(HttpClient http)
    {
        _http = http;
    }

    public async Task<JobSubmitResponse> SubmitGenerationAsync(
        GenerationRequest request, CancellationToken ct = default)
    {
        var payload = new
        {
            prompt = request.Prompt,
            negative_prompt = request.NegativePrompt,
            lora_paths = request.LoraPaths,
            lora_scales = request.LoraScales,
            reference_image_b64 = request.ReferenceImageB64,
            denoise_strength = request.DenoiseStrength,
            num_inference_steps = request.NumInferenceSteps,
            height = request.Height,
            width = request.Width,
            seed = request.Seed,
            guidance_scale = request.GuidanceScale,
        };

        var response = await _http.PostAsJsonAsync("/api/v1/generate", payload, ct);
        response.EnsureSuccessStatusCode();
        return (await response.Content.ReadFromJsonAsync<JobSubmitResponse>(cancellationToken: ct))!;
    }

    public async Task<JobStatusResponse?> GetJobStatusAsync(string jobId, CancellationToken ct = default)
    {
        return await _http.GetFromJsonAsync<JobStatusResponse>($"/api/v1/jobs/{jobId}", ct);
    }

    public async Task<Stream?> GetJobImageAsync(string jobId, CancellationToken ct = default)
    {
        var response = await _http.GetAsync($"/api/v1/jobs/{jobId}/image", ct);
        if (!response.IsSuccessStatusCode) return null;
        return await response.Content.ReadAsStreamAsync(ct);
    }

    public async Task<HealthResponse> HealthCheckAsync(CancellationToken ct = default)
    {
        return (await _http.GetFromJsonAsync<HealthResponse>("/health", ct))!;
    }
}
