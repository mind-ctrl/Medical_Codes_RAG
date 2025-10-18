$targets = @(
    "d:\\Power BI Project\\Ollama Project\\medrag_env",
    "d:\\Power BI Project\\Ollama Project\\medrag_env311",
    "d:\\Power BI Project\\Ollama Project\\chroma_data",
    "d:\\Power BI Project\\Ollama Project\\ICD10-CM Code Descriptions 2025"
)

foreach ($t in $targets) {
    Write-Output "================================================================"
    Write-Output "Target: $t"
    if (Test-Path $t) {
        $files = Get-ChildItem -Path $t -Recurse -File -ErrorAction SilentlyContinue
        $count = $files.Count
        $size = ($files | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / (1024*1024), 2)
        Write-Output "Files: $count    Total size (MB): $sizeMB"
        Write-Output "Top 10 largest files:"
        $files | Sort-Object Length -Descending | Select-Object -First 10 | ForEach-Object { Write-Output ("{0} bytes  {1}" -f $_.Length, $_.FullName) }
        Write-Output ""
        Write-Output "Sample first 20 files (paths):"
        $files | Select-Object -First 20 | ForEach-Object { Write-Output $_.FullName }
    }
    else {
        Write-Output "Directory not found: $t"
    }
}
Write-Output "================================================================"
