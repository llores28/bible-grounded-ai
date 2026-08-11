param(
    [ValidateSet("validate", "lint", "test", "pilot-drafts", "reviewer-readiness", "pilot-preflight", "preflight", "all")]
    [string]$Task = "all"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

function Invoke-Validate {
    python -m biblical_moral_ai --root $PSScriptRoot validate
    if ($LASTEXITCODE -ne 0) { throw "Structural validation failed." }
}

function Invoke-Lint {
    python -m ruff check (Join-Path $PSScriptRoot "src") (Join-Path $PSScriptRoot "tests/python")
    if ($LASTEXITCODE -ne 0) { throw "Python lint failed." }
}

function Invoke-Test {
    python -m unittest discover -s (Join-Path $PSScriptRoot "tests/python") -v
    if ($LASTEXITCODE -ne 0) { throw "Python tests failed." }
}

switch ($Task) {
    "validate" { Invoke-Validate }
    "lint" { Invoke-Lint }
    "test" { Invoke-Test }
    "preflight" {
        python -m biblical_moral_ai --root $PSScriptRoot preflight
        exit $LASTEXITCODE
    }
    "pilot-preflight" {
        python -m biblical_moral_ai --root $PSScriptRoot pilot-preflight
        exit $LASTEXITCODE
    }
    "pilot-drafts" {
        python -m biblical_moral_ai --root $PSScriptRoot audit-pilot-drafts
        exit $LASTEXITCODE
    }
    "reviewer-readiness" {
        python -m biblical_moral_ai --root $PSScriptRoot audit-reviewers
        exit $LASTEXITCODE
    }
    "all" {
        Invoke-Validate
        Invoke-Lint
        Invoke-Test
    }
}
