param(
    [string]$RepoName = "stellar-lc-platform",
    [ValidateSet("public", "private")]
    [string]$Visibility = "public",
    [string]$RemoteUrl = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
    git init
}

git add .

$hasCommit = git rev-parse --verify HEAD 2>$null
if (-not $hasCommit) {
    git commit -m "Initial project scaffold"
} else {
    $changes = git status --porcelain
    if ($changes) {
        git commit -m "Update project scaffold"
    }
}

$gh = Get-Command gh -ErrorAction SilentlyContinue

if ($gh) {
    gh auth status
    gh repo create $RepoName --$Visibility --source=. --remote=origin --push
    Write-Host "Published to GitHub."
    exit 0
}

if ($RemoteUrl) {
    $existingOrigin = git remote get-url origin 2>$null
    if ($existingOrigin) {
        git remote set-url origin $RemoteUrl
    } else {
        git remote add origin $RemoteUrl
    }

    git branch -M main
    git push -u origin main
    Write-Host "Published to GitHub."
    exit 0
}

Write-Host ""
Write-Host "Git is initialized and the first commit is ready."
Write-Host "GitHub CLI (gh) was not found and no -RemoteUrl was supplied."
Write-Host ""
Write-Host "Create an empty GitHub repository named '$RepoName', then rerun:"
Write-Host ".\scripts\publish_github.ps1 -RemoteUrl https://github.com/YOUR_USERNAME/$RepoName.git"
