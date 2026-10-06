$ErrorActionPreference='Stop'
$repo=Join-Path $PSScriptRoot '..\mt-kahypar'; $exe=Join-Path $repo 'build-gp00\mt-kahypar\application\MtKaHyPar.exe'
$input=(Join-Path $PSScriptRoot '..\graphs\er1024_p001_seed20260930.graph'); $runs=Join-Path $PSScriptRoot '..\runs'; $env:PATH='C:\mingw64\mingw64\bin;'+(Join-Path $repo 'build-gp00\gnu_14.2_cxx11_64_release')+';'+(Join-Path $PSScriptRoot '..\..\vcpkg\installed\x64-mingw-dynamic\bin')+';'+$env:PATH
$records=[System.Collections.Generic.List[object]]::new()
foreach($k in @(4,32)){
  $dir=Join-Path $runs "featurediag_er1024_s0_k${k}_ct20"; New-Item -ItemType Directory -Force -Path $dir|Out-Null
  Copy-Item -LiteralPath $input -Destination (Join-Path $dir 'input.graph') -Force
  $dump=Join-Path $dir 'model_features.csv'; $env:MKH_GP00_DUMP_FEATURES=$dump
  $args=@('-h','input.graph','--input-file-format=metis','--instance-type=graph','--preset-type=default','-k',"$k",'-e','0.03','-o','cut','-t','1','--seed','0','--c-t','20','--write-partition-file=true','--partition-output-folder','.')
  $cmd='"'+$exe+'" '+($args -join ' '); $watch=[Diagnostics.Stopwatch]::StartNew(); Push-Location $dir
  try{& $exe @args *> 'stdout.log';$exit=$LASTEXITCODE}finally{Pop-Location;$watch.Stop()}; $env:MKH_GP00_DUMP_FEATURES=''
  $part="input.graph.part$k.epsilon0.03.seed0.KaHyPar"; if($exit -ne 0 -or -not(Test-Path (Join-Path $dir $part)) -or -not(Test-Path $dump)){throw "probe failed k=$k exit=$exit"}
  $records.Add([pscustomobject]@{run=(Split-Path $dir -Leaf);k=$k;seed=0;graph='er1024_p001_seed20260930';extra_configuration='--c-t=20 (same for both k; diagnostic trigger only)';command=$cmd;exit_code=$exit;wall_seconds=[Math]::Round($watch.Elapsed.TotalSeconds,6);feature_rows=@((Import-Csv $dump)).Count;dump_sha256=(Get-FileHash -Algorithm SHA256 $dump).Hash})
  Write-Output "feature probe k=$k exit=$exit features=$((Import-Csv $dump).Count)"
}
$records|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $runs 'feature_probe_manifest.json') -Encoding utf8
