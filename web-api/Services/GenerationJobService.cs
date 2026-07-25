using System.Collections.Concurrent;
using MfluxExperiment.WebApi.Models;

namespace MfluxExperiment.WebApi.Services;

/// <summary>
/// In-memory job tracker on the web-api side.
/// Tracks jobs submitted by the frontend and maps them to
/// inference-service job IDs. In production, replace with
/// a database-backed queue (Postgres / Redis / Hangfire).
/// </summary>
public class GenerationJobService
{
    private readonly ConcurrentDictionary<Guid, JobRecord> _jobs = new();

    public record JobRecord
    {
        public Guid Id { get; init; } = Guid.NewGuid();
        public string InferenceJobId { get; init; } = "";
        public string Prompt { get; init; } = "";
        public string Status { get; set; } = "queued";
        public string? ResultUrl { get; set; }
        public string? Error { get; set; }
        public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
    }

    public JobRecord CreateJob(string inferenceJobId, string prompt)
    {
        var record = new JobRecord
        {
            InferenceJobId = inferenceJobId,
            Prompt = prompt,
        };
        _jobs[record.Id] = record;
        return record;
    }

    public JobRecord? GetJob(Guid id) => _jobs.GetValueOrDefault(id);

    public IEnumerable<JobRecord> GetAll() => _jobs.Values;

    public void UpdateFromInference(Guid id, JobStatusResponse status)
    {
        if (_jobs.TryGetValue(id, out var record))
        {
            record.Status = status.Status;
            record.ResultUrl = status.ResultUrl;
            record.Error = status.Error;
        }
    }
}
