using MfluxExperiment.WebApi.Models;

namespace MfluxExperiment.WebApi.Services;

/// <summary>
/// Abstraction over the inference backend.
/// Currently points at the local mflux FastAPI service.
/// Swap this implementation to talk to Replicate / Fal.ai / RunPod
/// without changing any calling code.
/// </summary>
public interface IInferenceClient
{
    Task<JobSubmitResponse> SubmitGenerationAsync(GenerationRequest request, CancellationToken ct = default);
    Task<JobStatusResponse?> GetJobStatusAsync(string jobId, CancellationToken ct = default);
    Task<Stream?> GetJobImageAsync(string jobId, CancellationToken ct = default);
    Task<HealthResponse> HealthCheckAsync(CancellationToken ct = default);
}
