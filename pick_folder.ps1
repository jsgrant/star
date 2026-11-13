# 폴더 고르기 창 (run_fov800.bat 이 사용) — 고른 경로를 한 줄로 출력
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = 'FOV 800 으로 변환할 환자 폴더(또는 imported 폴더)를 고르세요'
foreach ($p in @((Join-Path $env:LOCALAPPDATA 'Programs\RTDicomStudio\imported'),
                 'E:\Download(E_drv)\RTDicomStudio\RTDicomStudio\imported')) {
    if (Test-Path $p) { $d.SelectedPath = $p; break }
}
if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $d.SelectedPath }
