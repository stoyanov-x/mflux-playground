using Microsoft.AspNetCore.Mvc;
using MfluxExperiment.WebApi.Models;
using MfluxExperiment.WebApi.Services;

namespace MfluxExperiment.WebApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public class GenerationController : ControllerBase
{
    private readonly IInferenceClient _inference;
    private readonly GenerationJobService _jobs;

    public GenerationController(IInferenceClient inference, GenerationJobService jobs)
    {
        _inference = inference;
        _jobs = jobs;
    }

    /// <summary>
    /// Submit a new image generation job.
    /// Returns immediately; poll status via GET /api/generation/{id}.
    /// </summary>
    [HttpPost]
    public async Task<ActionResult<object>> Submit([FromBody] GenerationRequest request)
    {
        var inferenceResponse = await _inference.SubmitGenerationAsync(request);
        var record = await _jobs.CreateJobAsync(inferenceResponse.JobId, request.Prompt);

        return Accepted(new
        {
            id = record.Id,
            inferenceJobId = record.InferenceJobId,
            status = record.Status,
        });
    }

    /// <summary>
    /// Poll the status of a generation job.
    /// </summary>
    [HttpGet("{id:guid}")]
    public async Task<ActionResult<object>> GetStatus(Guid id)
    {
        var record = await _jobs.GetJobAsync(id);
        if (record is null) return NotFound();

        // Sync status from the inference service.
        var remote = await _inference.GetJobStatusAsync(record.InferenceJobId);
        if (remote is not null)
        {
            await _jobs.UpdateFromInferenceAsync(id, remote);
        }

        return Ok(new
        {
            id = record.Id,
            prompt = record.Prompt,
            status = record.Status,
            resultUrl = record.ResultUrl,
            error = record.Error,
            createdAt = record.CreatedAt,
        });
    }

    /// <summary>
    /// Download the generated image for a completed job.
    /// </summary>
    [HttpGet("{id:guid}/image")]
    public async Task<IActionResult> GetImage(Guid id)
    {
        var record = await _jobs.GetJobAsync(id);
        if (record is null) return NotFound();
        if (record.Status != "completed")
            return BadRequest("Job not yet completed");

        var imageStream = await _inference.GetJobImageAsync(record.InferenceJobId);
        if (imageStream is null) return NotFound();

        return File(imageStream, "image/png");
    }
}
