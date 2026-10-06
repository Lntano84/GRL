$ErrorActionPreference = 'Stop'
$OutDir = Join-Path $PSScriptRoot '..\graphs'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
function Write-MetisGraph([string]$Path, [System.Collections.Generic.List[int[]]]$Adj) {
  $edges = 0
  foreach ($row in $Adj) { $edges += $row.Length }
  if (($edges % 2) -ne 0) { throw 'Asymmetric adjacency count' }
  $lines = [System.Collections.Generic.List[string]]::new()
  $lines.Add("$($Adj.Count) $([int]($edges / 2))")
  foreach ($row in $Adj) { $lines.Add(($row | ForEach-Object { $_ + 1 }) -join ' ') }
  [IO.File]::WriteAllLines($Path, $lines, [Text.UTF8Encoding]::new($false))
  [pscustomobject]@{ path=$Path; nodes=$Adj.Count; undirected_edges=[int]($edges/2); sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash }
}
$n=1024
$grid=[System.Collections.Generic.List[int[]]]::new()
for ($v=0; $v -lt $n; $v++) {
  $r=[int][Math]::Floor($v/32); $c=$v%32
  $neighbors=[System.Collections.Generic.List[int]]::new()
  if ($r -gt 0) { $neighbors.Add($v-32) }
  if ($c -gt 0) { $neighbors.Add($v-1) }
  if ($c -lt 31) { $neighbors.Add($v+1) }
  if ($r -lt 31) { $neighbors.Add($v+32) }
  $grid.Add([int[]]$neighbors.ToArray())
}
$gridMeta=Write-MetisGraph (Join-Path $OutDir 'grid32x32_4n.graph') $grid
$seed=20260930; $p=0.01; $rng=[System.Random]::new($seed)
$erLists=[System.Collections.Generic.List[System.Collections.Generic.List[int]]]::new()
for ($i=0; $i -lt $n; $i++) { $erLists.Add([System.Collections.Generic.List[int]]::new()) }
for ($u=0; $u -lt $n; $u++) { for ($v=$u+1; $v -lt $n; $v++) { if ($rng.NextDouble() -lt $p) { $erLists[$u].Add($v); $erLists[$v].Add($u) } } }
$er=[System.Collections.Generic.List[int[]]]::new()
foreach ($list in $erLists) { $sorted=$list.ToArray(); [Array]::Sort($sorted); $er.Add([int[]]$sorted) }
$erMeta=Write-MetisGraph (Join-Path $OutDir 'er1024_p001_seed20260930.graph') $er
$metadata=[ordered]@{ generated_utc=(Get-Date).ToUniversalTime().ToString('o'); generator='PowerShell System.Random; iterate unordered pairs lexicographically u then v; include edge iff NextDouble()<p; undirected simple graph'; seed=$seed; p=$p; runtime=$PSVersionTable.PSVersion.ToString(); dotnet=[Environment]::Version.ToString(); graphs=@($gridMeta,$erMeta) }
$metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutDir 'input_manifest.json') -Encoding utf8
