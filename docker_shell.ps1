$ErrorActionPreference = "Stop"

$src = "research-agent"
$dst = "research-agent-gpu"
$img = "$src-snap:$(Get-Date -Format yyyyMMddHHmmss)"

$existing = docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $dst }
if ($existing) { docker rm -f $dst | Out-Null }

$c = docker inspect $src | ConvertFrom-Json | Select-Object -First 1
if (-not $c) { throw "container not found: $src" }

Write-Host "[1/3] snapshotting current container -> $img"
docker commit $src $img | Out-Null

Write-Host "[2/3] building create args from current container config"
$dockerArgs = New-Object System.Collections.Generic.List[string]

$dockerArgs.Add("create")
$dockerArgs.Add("--name");  $dockerArgs.Add($dst)
$dockerArgs.Add("--gpus");  $dockerArgs.Add("all")

if ($c.Config.OpenStdin) { $dockerArgs.Add("-i") }
if ($c.Config.Tty)       { $dockerArgs.Add("-t") }

if ($c.Config.User)       { $dockerArgs.Add("-u"); $dockerArgs.Add($c.Config.User) }
if ($c.Config.WorkingDir) { $dockerArgs.Add("-w"); $dockerArgs.Add($c.Config.WorkingDir) }

$oldId = [string]$c.Id
$oldShortId = if ($oldId.Length -ge 12) { $oldId.Substring(0,12) } else { $oldId }
if ($c.Config.Hostname -and $c.Config.Hostname -ne $oldId -and $c.Config.Hostname -ne $oldShortId) {
    $dockerArgs.Add("--hostname"); $dockerArgs.Add($c.Config.Hostname)
}

$rp = $c.HostConfig.RestartPolicy
if ($rp -and $rp.Name -and $rp.Name -ne "no") {
    $policy = $rp.Name
    if ($rp.Name -eq "on-failure" -and $rp.MaximumRetryCount -gt 0) {
        $policy = "$policy`:$($rp.MaximumRetryCount)"
    }
    $dockerArgs.Add("--restart"); $dockerArgs.Add($policy)
}

if ($c.HostConfig.Privileged)     { $dockerArgs.Add("--privileged") }
if ($c.HostConfig.ReadonlyRootfs) { $dockerArgs.Add("--read-only") }
if ($c.HostConfig.Init)           { $dockerArgs.Add("--init") }

if ($c.HostConfig.IpcMode -and $c.HostConfig.IpcMode -ne "private") {
    $dockerArgs.Add("--ipc"); $dockerArgs.Add($c.HostConfig.IpcMode)
}
if ($c.HostConfig.PidMode) {
    $dockerArgs.Add("--pid"); $dockerArgs.Add($c.HostConfig.PidMode)
}
if ($c.HostConfig.ShmSize -and $c.HostConfig.ShmSize -ne 67108864) {
    $dockerArgs.Add("--shm-size"); $dockerArgs.Add([string]$c.HostConfig.ShmSize)
}

$primaryNet = $c.HostConfig.NetworkMode
if ($primaryNet -and $primaryNet -ne "default" -and $primaryNet -ne "bridge") {
    $dockerArgs.Add("--network"); $dockerArgs.Add($primaryNet)
}

foreach ($e in @($c.Config.Env)) {
    if (-not $e) { continue }
    if ($e -like "HOSTNAME=*") { continue }
    $dockerArgs.Add("-e"); $dockerArgs.Add($e)
}

foreach ($d in @($c.HostConfig.Dns)) {
    if ($d) { $dockerArgs.Add("--dns"); $dockerArgs.Add($d) }
}
foreach ($d in @($c.HostConfig.DnsSearch)) {
    if ($d) { $dockerArgs.Add("--dns-search"); $dockerArgs.Add($d) }
}
foreach ($h in @($c.HostConfig.ExtraHosts)) {
    if ($h) { $dockerArgs.Add("--add-host"); $dockerArgs.Add($h) }
}
foreach ($x in @($c.HostConfig.CapAdd)) {
    if ($x) { $dockerArgs.Add("--cap-add"); $dockerArgs.Add($x) }
}
foreach ($x in @($c.HostConfig.CapDrop)) {
    if ($x) { $dockerArgs.Add("--cap-drop"); $dockerArgs.Add($x) }
}
foreach ($g in @($c.HostConfig.GroupAdd)) {
    if ($g) { $dockerArgs.Add("--group-add"); $dockerArgs.Add($g) }
}
foreach ($s in @($c.HostConfig.SecurityOpt)) {
    if ($s) { $dockerArgs.Add("--security-opt"); $dockerArgs.Add($s) }
}
if ($c.HostConfig.Sysctls) {
    foreach ($p in $c.HostConfig.Sysctls.PSObject.Properties) {
        $dockerArgs.Add("--sysctl"); $dockerArgs.Add("$($p.Name)=$($p.Value)")
    }
}

foreach ($u in @($c.HostConfig.Ulimits)) {
    if (-not $u) { continue }
    $dockerArgs.Add("--ulimit")
    if ($u.Soft -ne $null -and $u.Hard -ne $null) {
        $dockerArgs.Add("$($u.Name)=$($u.Soft):$($u.Hard)")
    } else {
        $dockerArgs.Add("$($u.Name)")
    }
}

foreach ($d in @($c.HostConfig.Devices)) {
    if (-not $d) { continue }
    $perm = if ($d.CgroupPermissions) { $d.CgroupPermissions } else { "rwm" }
    $dockerArgs.Add("--device")
    $dockerArgs.Add("$($d.PathOnHost):$($d.PathInContainer):$perm")
}
foreach ($r in @($c.HostConfig.DeviceCgroupRules)) {
    if ($r) { $dockerArgs.Add("--device-cgroup-rule"); $dockerArgs.Add($r) }
}

foreach ($m in @($c.Mounts)) {
    if (-not $m) { continue }
    switch ($m.Type) {
        "bind" {
            $opt = "type=bind,src=$($m.Source),dst=$($m.Destination)"
            if (-not $m.RW) { $opt += ",readonly" }
            if ($m.Propagation) { $opt += ",bind-propagation=$($m.Propagation)" }
            $dockerArgs.Add("--mount"); $dockerArgs.Add($opt)
        }
        "volume" {
            $srcName = if ($m.Name) { $m.Name } else { $m.Source }
            $opt = "type=volume,src=$srcName,dst=$($m.Destination)"
            if (-not $m.RW) { $opt += ",readonly" }
            $dockerArgs.Add("--mount"); $dockerArgs.Add($opt)
        }
        "tmpfs" {
            $dockerArgs.Add("--tmpfs"); $dockerArgs.Add($m.Destination)
        }
    }
}

if ($c.HostConfig.PortBindings) {
    foreach ($p in $c.HostConfig.PortBindings.PSObject.Properties) {
        $containerPort = $p.Name
        foreach ($b in @($p.Value)) {
            if (-not $b -or -not $b.HostPort) { continue }
            $pub = if ($b.HostIp -and $b.HostIp -ne "0.0.0.0" -and $b.HostIp -ne "::") {
                "$($b.HostIp):$($b.HostPort):$containerPort"
            } else {
                "$($b.HostPort):$containerPort"
            }
            $dockerArgs.Add("-p"); $dockerArgs.Add($pub)
        }
    }
}

$dockerArgs.Add($img)

Write-Host "[3/3] creating new GPU container -> $dst"
docker @dockerArgs

Write-Host ""
Write-Host "Created: $dst"
Write-Host "Old container untouched: $src"