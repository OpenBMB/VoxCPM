<#
.SYNOPSIS
    Les deux transferts d'une session de narration sur GPU loué.

.DESCRIPTION
    Un GPU loué à l'heure est créé pour un livre puis détruit. Il manque donc,
    à chaque fois, exactement deux choses que `git clone` ne fournit pas :

      - les références des voix clonées, `assets/voices/`, volontairement hors
        du dépôt parce que ce sont des enregistrements de personnes réelles et
        que le dépôt est public ;
      - le livre à narrer.

    Et à la fin, il faut rapatrier les chapitres avant de détruire la machine.

    `rsync`, que documentait CLOUD.md, n'existe pas sur Windows. `scp` si —
    il est livré avec OpenSSH depuis Windows 10 — et suffit largement pour
    4,6 Mo de références et quelques centaines de Mo de chapitres.

.PARAMETER Action
    push    envoie les voix clonées (et, avec -Book, le livre à narrer)
    pull    rapatrie les chapitres produits dans output/
    tunnel  ouvre l'interface Gradio distante sur http://127.0.0.1:8808
            sans exposer le port de la machine louée

.EXAMPLE
    ./scripts/gpu_session.ps1 push -RemoteHost 194.26.196.4 -Port 22077 -Book 'C:\livres\autour_de_la_lune.epub'
    ./scripts/gpu_session.ps1 tunnel -RemoteHost 194.26.196.4 -Port 22077
    ./scripts/gpu_session.ps1 pull -RemoteHost 194.26.196.4 -Port 22077 -Name autour_de_la_lune
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet('push', 'pull', 'tunnel')]
    [string]$Action,

    [Parameter(Mandatory)]
    [string]$RemoteHost,

    [int]$Port = 22,
    [string]$User = 'root',

    # Là où cloud_setup.sh installe quand il trouve un volume persistant.
    [string]$RemoteDir = '/workspace/voxcpm',

    # push : le livre à narrer. pull : le nom du dossier sous output/.
    [string]$Book,
    [string]$Name,

    [string]$IdentityFile,
    [int]$LocalPort = 8808
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
    throw "scp introuvable. Installez OpenSSH : Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"
}

$repo = Split-Path $PSScriptRoot -Parent
$target = "$User@$RemoteHost"

# Les options communes, construites une fois. -P pour scp, -p pour ssh : la
# différence de casse est une vraie source d'erreurs, elle est isolée ici.
$scpOpts = @('-P', $Port)
$sshOpts = @('-p', $Port)
if ($IdentityFile) {
    $scpOpts += @('-i', $IdentityFile)
    $sshOpts += @('-i', $IdentityFile)
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    Write-Host "  $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$What a échoué (code $LASTEXITCODE)." }
}

switch ($Action) {

    'push' {
        $voices = Join-Path $repo 'assets\voices'
        if (-not (Test-Path $voices)) { throw "Introuvable : $voices" }

        $wavs = Get-ChildItem $voices -Filter *.wav
        $mo = [math]::Round((($wavs | Measure-Object Length -Sum).Sum) / 1MB, 1)
        Write-Host "Voix clonées : $($wavs.Count) fichiers, $mo Mo" -ForegroundColor Cyan

        # Le dossier distant existe (il vient du dépôt), mais pas forcément si
        # l'installation a été déplacée — et scp ne le crée pas.
        Invoke-Checked ssh ($sshOpts + @($target, "mkdir -p '$RemoteDir/assets/voices'")) 'mkdir distant'
        Invoke-Checked scp ($scpOpts + @('-r', "$voices\*", "${target}:$RemoteDir/assets/voices/")) 'Envoi des voix'

        # preset_voices.json est versionné, donc déjà à jour côté distant après
        # un clone — mais pas si vous venez d'ajouter une voix sans committer.
        $conf = Join-Path $repo 'conf\preset_voices.json'
        Invoke-Checked scp ($scpOpts + @($conf, "${target}:$RemoteDir/conf/preset_voices.json")) 'Envoi de preset_voices.json'

        if ($Book) {
            if (-not (Test-Path $Book)) { throw "Livre introuvable : $Book" }
            $leaf = Split-Path $Book -Leaf
            Invoke-Checked scp ($scpOpts + @($Book, "${target}:$RemoteDir/$leaf")) 'Envoi du livre'
            Write-Host ""
            Write-Host "Sur la machine louée :" -ForegroundColor Green
            Write-Host @"
  source $RemoteDir/env.sh
  nohup python scripts/narrate_book.py '$leaf' --device cuda \
      --voice 'Aurore — livre audio' --assemble m4b --export-acx \
      > narration.log 2>&1 &
"@
        }
    }

    'pull' {
        if (-not $Name) { throw "Précisez -Name <nom du livre> (le dossier sous output/)." }

        $localOut = Join-Path $repo 'output'
        if (-not (Test-Path $localOut)) { New-Item -ItemType Directory $localOut | Out-Null }

        $remote = "$RemoteDir/output/book_$Name"
        Invoke-Checked ssh ($sshOpts + @($target, "test -d '$remote'")) "Le dossier distant $remote n'existe pas ; vérifiez -Name"
        Invoke-Checked scp ($scpOpts + @('-r', "${target}:$remote", $localOut)) 'Rapatriement'

        $got = Join-Path $localOut "book_$Name"
        $n = (Get-ChildItem $got -Recurse -File | Measure-Object).Count
        Write-Host ""
        Write-Host "$n fichiers dans $got" -ForegroundColor Green
        Write-Host "Vérifiez avant de détruire la machine — elle est facturée tant qu'elle existe." -ForegroundColor Yellow
    }

    'tunnel' {
        Write-Host "Tunnel ouvert : http://127.0.0.1:$LocalPort" -ForegroundColor Green
        Write-Host "Rien n'est exposé sur la machine louée. Ctrl+C pour fermer." -ForegroundColor DarkGray
        & ssh @sshOpts -N -L "${LocalPort}:127.0.0.1:8808" $target
    }
}
