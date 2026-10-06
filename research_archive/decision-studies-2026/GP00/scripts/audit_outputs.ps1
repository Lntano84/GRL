$ErrorActionPreference='Stop'
$Base=Join-Path $PSScriptRoot '..'
$Runs=Join-Path $Base 'runs'
$Graphs=Join-Path $Base 'graphs'
$Out=Join-Path $Base 'outputs'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$manifest=Get-Content -Raw (Join-Path $Runs 'run_manifest.json') | ConvertFrom-Json
$graphMap=@{ grid32x32_4n=(Join-Path $Graphs 'grid32x32_4n.graph'); er1024_p001_seed20260930=(Join-Path $Graphs 'er1024_p001_seed20260930.graph') }
$results=[System.Collections.Generic.List[object]]::new()
foreach($r in $manifest){
  $inputPath=$graphMap[$r.graph]
  $lines=Get-Content -LiteralPath $inputPath
  $head=$lines[0] -split '\s+'
  $n=[int]$head[0]; $m=[int]$head[1]
  $adj=[System.Collections.Generic.List[int[]]]::new()
  $directed=0; $arcs=[System.Collections.Generic.HashSet[string]]::new()
  for($u=0;$u -lt $n;$u++){
    $tokens=@(); if($lines[$u+1].Trim().Length -gt 0){$tokens=$lines[$u+1].Trim() -split '\s+'}
    $row=[int[]]@($tokens | ForEach-Object { [int]$_ - 1 })
    $adj.Add($row)
    foreach($v in $row){$directed++; [void]$arcs.Add("$u`:$v")}
  }
  $symmetryErrors=0
  foreach($arc in $arcs){$parts=$arc -split ':'; if(-not $arcs.Contains("$($parts[1])`:$($parts[0])")){$symmetryErrors++}}
  if($symmetryErrors -ne 0 -or $directed -ne 2*$m){throw "Bad METIS input audit for $($r.graph): directed=$directed m=$m asym=$symmetryErrors"}
  $partPath=Join-Path (Join-Path $Runs $r.run) $r.partition_file
  $partLines=Get-Content -LiteralPath $partPath
  if($partLines.Count -ne $n){throw "Partition row count mismatch: $($r.run)"}
  $parts=[int[]]@($partLines | ForEach-Object { [int]$_.Trim() })
  $counts=[int[]]::new($r.k)
  $invalidLabels=0; $cut=0
  for($u=0;$u -lt $n;$u++){
    $b=$parts[$u]; if($b -lt 0 -or $b -ge $r.k){$invalidLabels++; continue}; $counts[$b]++
    foreach($v in $adj[$u]){if($u -lt $v -and $parts[$u] -ne $parts[$v]){$cut++}}
  }
  $maxAllowed=[int][Math]::Floor((1.0+0.03)*$n/$r.k+1e-10)
  $log=Get-Content -Raw (Join-Path (Join-Path $Runs $r.run) 'stdout.log')
  $cutMatches=[regex]::Matches($log,'cut\s*=\s*(\d+)\s+\(primary objective function\)')
  if($cutMatches.Count -lt 1){throw "Could not parse final cut from $($r.run)"}
  $solverCut=[int]$cutMatches[$cutMatches.Count-1].Groups[1].Value
  $timeMatch=[regex]::Match($log,'Partitioning Time\s*=\s*([0-9.eE+-]+)\s*s')
  $solverTime=$null; if($timeMatch.Success){$solverTime=[double]::Parse($timeMatch.Groups[1].Value,[Globalization.CultureInfo]::InvariantCulture)}
  $externalWall=$null; if($r.time_source -eq 'external_wall_clock'){$externalWall=[double]$r.wall_seconds}
  $results.Add([pscustomobject]@{run=$r.run; method=$r.method; graph=$r.graph; k=$r.k; seed=$r.seed; cut_independent=$cut; cut_solver=$solverCut; cut_match=($cut -eq $solverCut); min_block=($counts | Measure-Object -Minimum).Minimum; max_block=($counts | Measure-Object -Maximum).Maximum; block_sizes=($counts -join ','); max_allowed=$maxAllowed; balance_feasible=(($invalidLabels -eq 0) -and (($counts | Measure-Object -Minimum).Minimum -gt 0) -and (($counts | Measure-Object -Maximum).Maximum -le $maxAllowed)); invalid_labels=$invalidLabels; solver_partitioning_seconds=$solverTime; external_wall_seconds=$externalWall; elapsed_source=$r.time_source; input_sha256=$r.input_sha256; exit_code=$r.exit_code})
}
$results | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $Out 'run_results.csv')

$featureSummary=[System.Collections.Generic.List[object]]::new()
foreach($grp in ($manifest | Where-Object method -eq 'learned' | Group-Object graph,seed)){
  $items=@($grp.Group | Sort-Object k)
  $lo=$items | Where-Object k -eq 4 | Select-Object -First 1
  $hi=$items | Where-Object k -eq 32 | Select-Object -First 1
  $loPath=Join-Path (Join-Path $Runs $lo.run) $lo.feature_dump
  $hiPath=Join-Path (Join-Path $Runs $hi.run) $hi.feature_dump
  if(-not (Test-Path $loPath) -or -not (Test-Path $hiPath)){throw "Missing k comparison dump for $($grp.Name)"}
  $a=@(Import-Csv -LiteralPath $loPath); $b=@(Import-Csv -LiteralPath $hiPath)
  if($a.Count -ne $b.Count){throw "Feature edge count differs for $($grp.Name): $($a.Count) vs $($b.Count)"}
  $rawCols=@(0..31 | ForEach-Object { 'raw_f{0:d2}' -f $_ }); $normCols=@(0..31 | ForEach-Object { 'normalized_f{0:d2}' -f $_ })
  $rawChangedRows=0; $rawChangedCells=0; $normChangedRows=0; $normChangedCells=0; $predChanged=0
  $maxRaw=0.0; $maxNorm=0.0; $maxPred=0.0; $alignErrors=0
  for($i=0;$i -lt $a.Count;$i++){
    if($a[$i].edge_id -ne $b[$i].edge_id -or $a[$i].source -ne $b[$i].source -or $a[$i].target -ne $b[$i].target){$alignErrors++;continue}
    $rraw=0;$rnorm=0
    foreach($col in $rawCols){$d=[Math]::Abs([double]::Parse($a[$i].$col,[Globalization.CultureInfo]::InvariantCulture)-[double]::Parse($b[$i].$col,[Globalization.CultureInfo]::InvariantCulture));if($d -gt 0){$rawChangedCells++;$rraw++};if($d -gt $maxRaw){$maxRaw=$d}}
    foreach($col in $normCols){$d=[Math]::Abs([double]::Parse($a[$i].$col,[Globalization.CultureInfo]::InvariantCulture)-[double]::Parse($b[$i].$col,[Globalization.CultureInfo]::InvariantCulture));if($d -gt 0){$normChangedCells++;$rnorm++};if($d -gt $maxNorm){$maxNorm=$d}}
    if($rraw -gt 0){$rawChangedRows++}; if($rnorm -gt 0){$normChangedRows++}
    $pd=[Math]::Abs([double]::Parse($a[$i].prediction,[Globalization.CultureInfo]::InvariantCulture)-[double]::Parse($b[$i].prediction,[Globalization.CultureInfo]::InvariantCulture));if($pd -gt 0){$predChanged++};if($pd -gt $maxPred){$maxPred=$pd}
  }
  # Each original undirected edge should appear as two directed arcs with the same canonicalized input and score.
  $map=@{}; foreach($row in $a){$map["$($row.source):$($row.target)"]=$row}
  $reciprocalPairs=0;$reciprocalMissing=0;$reciprocalFeatureMismatches=0;$reciprocalMaxScoreDiff=0.0
  foreach($row in $a){if([int]$row.source -lt [int]$row.target){$rev=$map["$($row.target):$($row.source)"];if($null -eq $rev){$reciprocalMissing++}else{$reciprocalPairs++;foreach($col in $rawCols){if([double]$row.$col -ne [double]$rev.$col){$reciprocalFeatureMismatches++;break}};$sd=[Math]::Abs([double]$row.prediction-[double]$rev.prediction);if($sd -gt $reciprocalMaxScoreDiff){$reciprocalMaxScoreDiff=$sd}}}}
  $featureSummary.Add([pscustomobject]@{graph=$lo.graph; seed=$lo.seed; k4_edge_rows=$a.Count; k32_edge_rows=$b.Count; alignment_errors=$alignErrors; raw_feature_rows_changed=$rawChangedRows; raw_feature_cells_changed=$rawChangedCells; raw_feature_cells_total=($a.Count*32); max_abs_raw_delta=$maxRaw; normalized_feature_rows_changed=$normChangedRows; normalized_feature_cells_changed=$normChangedCells; max_abs_normalized_delta=$maxNorm; predictions_changed=$predChanged; max_abs_prediction_delta=$maxPred; reciprocal_undirected_edges=$reciprocalPairs; reciprocal_missing=$reciprocalMissing; reciprocal_raw_vector_mismatches=$reciprocalFeatureMismatches; reciprocal_max_abs_prediction_delta=$reciprocalMaxScoreDiff; skip_comm_1_k4=$a[0].skip_comm_1; skip_comm_1_k32=$b[0].skip_comm_1})
}
$featureSummary | Export-Csv -NoTypeInformation -Encoding utf8 -LiteralPath (Join-Path $Out 'k_conditioning.csv')
$summary=[ordered]@{run_count=$results.Count; all_exits_zero=(($results | Where-Object exit_code -ne 0).Count -eq 0); all_cuts_match=(($results | Where-Object cut_match -ne $true).Count -eq 0); all_balanced=(($results | Where-Object balance_feasible -ne $true).Count -eq 0); graph_input_audits='Both graph files are undirected, symmetric, have 1024 nodes, and their listed undirected edge counts match the headers.'; feature_pairs=$featureSummary; result_csv=(Join-Path $Out 'run_results.csv'); conditioning_csv=(Join-Path $Out 'k_conditioning.csv')}
$summary | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $Out 'audit.json') -Encoding utf8
$results | Format-Table method,graph,k,seed,cut_independent,balance_feasible,solver_partitioning_seconds,external_wall_seconds -AutoSize
$featureSummary | Format-Table graph,seed,k4_edge_rows,raw_feature_rows_changed,raw_feature_cells_changed,max_abs_raw_delta,predictions_changed,max_abs_prediction_delta,reciprocal_missing,reciprocal_raw_vector_mismatches -AutoSize

