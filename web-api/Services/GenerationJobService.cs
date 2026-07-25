using Microsoft.EntityFrameworkCore;
using MfluxExperiment.WebApi.Data;
using MfluxExperiment.WebApi.Models;

namespace MfluxExperiment.WebApi.Services;

/// <summary>
/// SQLite-backed job tracker on the web-api side.
/// Persists jobs so they survive restarts. Swap for Postgres
/// / Redis / Hangfire in production deployments.
/// </summary>
public class GenerationJobService(AppDbContext db)
{
    public async Task<JobRecord> CreateJobAsync(string inferenceJobId, string prompt)
    {
        var record = new JobRecord
        {
            Id = Guid.NewGuid(),
            InferenceJobId = inferenceJobId,
            Prompt = prompt,
            Status = "queued",
            CreatedAt = DateTime.UtcNow,
        };
        db.Jobs.Add(record);
        await db.SaveChangesAsync();
        return record;
    }

    public async Task<JobRecord?> GetJobAsync(Guid id)
    {
        return await db.Jobs.FindAsync(id);
    }

    public async Task<List<JobRecord>> GetAllAsync()
    {
        return await db.Jobs
            .OrderByDescending(j => j.CreatedAt)
            .Take(50)
            .ToListAsync();
    }

    public async Task UpdateFromInferenceAsync(Guid id, JobStatusResponse status)
    {
        var record = await db.Jobs.FindAsync(id);
        if (record is null) return;

        record.Status = status.Status;
        record.ResultUrl = status.ResultUrl;
        record.Error = status.Error;
        await db.SaveChangesAsync();
    }
}
