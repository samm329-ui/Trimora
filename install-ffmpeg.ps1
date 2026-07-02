$ffmpegDir = Join-Path $PSScriptRoot "ffmpeg"
$binDir = Join-Path $ffmpegDir "bin"
$ffmpegExe = Join-Path $binDir "ffmpeg.exe"

if (Test-Path $ffmpegExe) {
    Write-Host "FFmpeg already installed locally at $binDir"
    return $binDir
}

$url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$zip = Join-Path $env:TEMP "ffmpeg-release-essentials.zip"

Write-Host "Downloading FFmpeg..."
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
} catch {
    Write-Host "Download failed: $_"
    exit 1
}

Write-Host "Extracting..."
try {
    $tempExtract = Join-Path $env:TEMP "ffmpeg-extract"
    if (Test-Path $tempExtract) { Remove-Item -Recurse -Force $tempExtract }
    Expand-Archive -Path $zip -DestinationPath $tempExtract -Force
    $extractedDir = Get-ChildItem -Path $tempExtract -Directory | Select-Object -First 1
    $srcBin = Join-Path $extractedDir.FullName "bin"
    if (Test-Path $srcBin) {
        if (Test-Path $ffmpegDir) { Remove-Item -Recurse -Force $ffmpegDir }
        Move-Item -Path $srcBin -Destination $binDir -Force
        Write-Host "FFmpeg extracted to $binDir"
    } else {
        Write-Host "Unexpected archive structure"
        exit 1
    }
} catch {
    Write-Host "Extraction failed: $_"
    exit 1
} finally {
    Remove-Item -Force $zip -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $tempExtract -ErrorAction SilentlyContinue
}

if (Test-Path $ffmpegExe) {
    Write-Host "FFmpeg installed: $ffmpegExe"
    return $binDir
} else {
    Write-Host "Installation failed: ffmpeg.exe not found"
    exit 1
}
