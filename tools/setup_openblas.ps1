param([string]$MsysRoot = 'C:\msys64')

# Download/extract signed MSYS2 UCRT64 packages locally; do not install or
# upgrade anything in the user's MSYS2 installation.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pacman = Join-Path $MsysRoot 'usr/bin/pacman.exe'
if (-not (Test-Path -LiteralPath $pacman)) {
    throw "MSYS2 pacman was not found at $pacman. Pass -MsysRoot for your installation."
}

$cache = Join-Path $projectRoot 'build/deps/cache'
$destination = Join-Path $projectRoot 'build/deps/openblas'
New-Item -ItemType Directory -Force $cache, $destination | Out-Null
$package = 'mingw-w64-ucrt-x86_64-openblas'
$archives = @(& $pacman -Sp --needed --print-format '%f' $package)
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve OpenBLAS packages from the MSYS2 package database.' }
if ($archives.Count -eq 0) {
    Write-Output "OpenBLAS is already installed. Prefix: $MsysRoot/ucrt64"
    return
}

& $pacman -Sw --needed --noconfirm --cachedir $cache.Replace('\', '/') $package
if ($LASTEXITCODE -ne 0) { throw 'Downloading or verifying OpenBLAS packages failed.' }
foreach ($archive in $archives) {
    if ($archive -notmatch '^mingw-w64-ucrt-x86_64-[^/\\]+\.pkg\.tar\.(zst|xz)$') {
        throw "Unexpected package filename: $archive"
    }
    $archivePath = Join-Path $cache $archive
    if (-not (Test-Path -LiteralPath $archivePath)) { throw "Missing downloaded package: $archivePath" }
    & tar -xf $archivePath -C $destination
    if ($LASTEXITCODE -ne 0) { throw "Extracting $archive failed." }
}
Write-Output "OpenBLAS prefix: $destination/ucrt64"
Write-Output 'Run nob test next; CMake discovers this local prefix automatically with MinGW.'
