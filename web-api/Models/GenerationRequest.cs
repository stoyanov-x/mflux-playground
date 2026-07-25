namespace MfluxExperiment.WebApi.Models;

/// <summary>
/// Matches the inference service contract so the web API can
/// forward requests without mapping.
/// </summary>
public record GenerationRequest(
    string Prompt,
    string? NegativePrompt = null,
    List<string>? LoraPaths = null,
    List<double>? LoraScales = null,
    string? ReferenceImageB64 = null,
    double DenoiseStrength = 0.75,
    int? NumInferenceSteps = null,
    int? Height = null,
    int? Width = null,
    int? Seed = null,
    double GuidanceScale = 3.5
);

public record JobStatusResponse(
    string JobId,
    string Status,
    double Progress = 0.0,
    string? ResultUrl = null,
    string? Error = null,
    int? SeedUsed = null,
    double? ElapsedSeconds = null
);

public record JobSubmitResponse(string JobId, string Status, string Message = "Job accepted");

public record HealthResponse(string Status, string ModelAlias, List<string> LoraLoaded, int? Quantize);
