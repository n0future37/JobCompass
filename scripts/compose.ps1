param([ValidateSet('build','up','down','check','status')][string]$Action='check')
$ErrorActionPreference='Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$priorRedis = $env:REDIS_PASSWORD
$priorSili = $env:SILI_API_KEY
try {
    if (-not $env:SILI_API_KEY) {
        $env:SILI_API_KEY = [Environment]::GetEnvironmentVariable('SILI_API_KEY', 'User')
    }
    # 仅读取既有本机配置到当前进程，不输出密码，不修改公共配置。
    $entry = Get-Content -LiteralPath (Join-Path $projectRoot '.env.local') | Where-Object { $_ -match '^REDIS_PASSWORD=' } | Select-Object -First 1
    if (-not $entry) { throw '本机.env.local缺少REDIS_PASSWORD' }
    $env:REDIS_PASSWORD = ($entry -split '=',2)[1].Trim().Trim('"').Trim("'")
    Push-Location $projectRoot
    try {
        switch ($Action) {
            'check' { & docker compose --env-file .env.local config --quiet }
            'status' { & docker compose --env-file .env.local ps }
            'build' { & docker compose --env-file .env.local build }
            'up' { & docker compose --env-file .env.local up -d }
            'down' { & docker compose --env-file .env.local down }
        }
        if ($LASTEXITCODE -ne 0) { throw "JobCompass compose操作失败，退出码$LASTEXITCODE" }
    } finally { Pop-Location }
} finally {
    $env:REDIS_PASSWORD = $priorRedis
    $env:SILI_API_KEY = $priorSili
}
