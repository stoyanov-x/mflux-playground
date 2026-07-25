using Microsoft.EntityFrameworkCore;
using MfluxExperiment.WebApi.Data;
using MfluxExperiment.WebApi.Services;

var builder = WebApplication.CreateBuilder(args);

// ── Database ────────────────────────────────────────────────

var dbPath = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "data", "mflux_jobs.db");
Directory.CreateDirectory(Path.GetDirectoryName(dbPath)!);

builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlite($"Data Source={dbPath}"));

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

// SQLite-backed job tracker (swap for Postgres/Redis in production).
builder.Services.AddScoped<GenerationJobService>();

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

// Auto-create tables on startup.
using (var scope = app.Services.CreateScope())
{
    var ctx = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    ctx.Database.EnsureCreated();
}

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors();
app.MapControllers();

app.Run();
