$ErrorActionPreference = 'Stop'
$Repo = Join-Path $PSScriptRoot '..\mt-kahypar'
$Exe = Join-Path $Repo 'build-gp00\mt-kahypar\application\MtKaHyPar.exe'
$Graphs = Join-Path $PSScriptRoot '..\graphs'
$Runs = Join-Path $PSScriptRoot '..\runs'
$env:PATH = 'C:\mingw64\mingw64\bin;' + (Join-Path $Repo 'build-gp00\gnu_14.2_cxx11_64_release') + ';' + (Join-Path $PSScriptRoot '..\..\vcpkg\installed\x64-mingw-dynamic\bin') + ';' + $env:PATH
$cases = @(
  @{ name='grid32x32_4n'; path=(Join-Path $Graphs 'grid32x32_4n.graph') },
  @{ name='er1024_p001_seed20260930'; path=(Join-Path $Graphs 'er1024_p001_seed20260930.graph') }
)
$results = [System.Collections.Generic.List[object]]::new()
foreach ($case in $cases) {
  foreach ($k in @(4,32)) {
    foreach ($seed in @(0,1)) {
      foreach ($method in @('learned','baseline')) {
        $name = "$($case.name)_k${k}_s${seed}_$method"
        $dir = Join-Path $Runs $name
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $inputCopy=Join-Path $dir 'input.graph'
        if (-not (Test-Path $inputCopy)) { Copy-Item -LiteralPath $case.path -Destination $inputCopy }
        $inputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $inputCopy).Hash
        $args = @('-h','input.graph','--input-file-format=metis','--instance-type=graph','--preset-type=default','-k',"$k",'-e','0.03','-o','cut','-t','1','--seed',"$seed",'--write-partition-file=true','--partition-output-folder','.')
        if ($method -eq 'baseline') { $args += @('--c-guiding-by-integrated-model=false','--c-rating-degree-similarity-policy=always_accept') }
        $cmd = '"' + $Exe + '" ' + (($args | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' ')
        $partName = "input.graph.part$k.epsilon0.03.seed$seed.KaHyPar"
        $partPath = Join-Path $dir $partName
        $featurePath=Join-Path $dir 'model_features.csv'
        if (Test-Path $partPath) {
          $logText=Get-Content -Raw (Join-Path $dir 'stdout.log')
          $mt=[regex]::Match($logText,'Partitioning Time\s*=\s*([0-9.eE+-]+)\s*s')
          $wall=$null; $timeSource='recovered_solver_reported_time'
          if ($mt.Success) { $wall=[double]::Parse($mt.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture) }
          $exit=0
        } else {
          $env:MKH_GP00_DUMP_FEATURES = ''
          if ($method -eq 'learned') { $env:MKH_GP00_DUMP_FEATURES = $featurePath }
          $watch = [Diagnostics.Stopwatch]::StartNew()
          Push-Location $dir
          try { & $Exe @args *> 'stdout.log'; $exit = $LASTEXITCODE }
          finally { Pop-Location; $watch.Stop() }
          $env:MKH_GP00_DUMP_FEATURES = ''
          $wall=[Math]::Round($watch.Elapsed.TotalSeconds,6); $timeSource='external_wall_clock'
        }
        $featureName=$null; if (Test-Path $featurePath) { $featureName='model_features.csv' }
        if ($exit -ne 0 -or -not (Test-Path $partPath)) { throw "Run failed: $name (exit=$exit)" }
        $results.Add([pscustomobject]@{run=$name; method=$method; graph=$case.name; k=$k; seed=$seed; input_sha256=$inputHash; command=$cmd; exit_code=$exit; wall_seconds=$wall; time_source=$timeSource; partition_file=$partName; feature_dump=$featureName})
        Write-Output "$name exit=$exit time=$wall src=$timeSource features=$featureName"
      }
    }
  }
}
$results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Runs 'run_manifest.json') -Encoding utf8
