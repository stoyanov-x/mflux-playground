using Microsoft.EntityFrameworkCore;

namespace MfluxExperiment.WebApi.Data;

public class AppDbContext(DbContextOptions<AppDbContext> options) : DbContext(options)
{
    public DbSet<JobRecord> Jobs => Set<JobRecord>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<JobRecord>(entity =>
        {
            entity.ToTable("web_jobs");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Id).ValueGeneratedOnAdd();
            entity.Property(e => e.Prompt).IsRequired();
            entity.Property(e => e.Status).IsRequired().HasMaxLength(20);
            entity.HasIndex(e => e.Status);
            entity.HasIndex(e => e.CreatedAt).IsDescending();
        });
    }
}

public class JobRecord
{
    public Guid Id { get; set; }
    public string InferenceJobId { get; set; } = "";
    public string Prompt { get; set; } = "";
    public string Status { get; set; } = "queued";
    public string? ResultUrl { get; set; }
    public string? Error { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
