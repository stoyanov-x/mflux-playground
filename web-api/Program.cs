using MfluxExperiment.WebApi.Services;

var builder = WebApplication.CreateBuilder(args);

// ── Services ────────────────────────────────────────────────

builder.Services.AddControllers();
builder.Services.AddOpenApi();

// Inference client — swap this registration when moving to a cloud target.
builder.Services.AddHttpClient<IInferenceClient, MfluxInferenceClient>(client =>
{
    client.BaseAddress = new Uri(
        builder.Configuration.GetValue<string>("InferenceBaseUrl")
        ?? "http://localhost:8000");
    client.Timeout = TimeSpan.FromMinutes(5); // generations can be slow
});

// In-memory job tracker (swap for DB-backed in production).
builder.Services.AddSingleton<GenerationJobService>();

// ── CORS ────────────────────────────────────────────────────

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

// ── Pipeline ────────────────────────────────────────────────

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors();
app.MapControllers();

app.Run();
