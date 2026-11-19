$ErrorActionPreference = "Stop"

python scripts\make_train_configs.py

Get-ChildItem -LiteralPath configs\generated -Filter *.yaml |
    Sort-Object Name |
    ForEach-Object {
        Write-Host "Training $($_.Name)"
        llamafactory-cli train $_.FullName
    }

