@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.29 > nul 2>&1
echo === INCLUDE first entry ===
for /f "tokens=1 delims=;" %%a in ("%INCLUDE%") do echo %%a
echo.
echo === which cl ===
where cl
echo.
echo === nvcc would call cl with this env. Direct nvcc probe: ===
nvcc --version
echo.
echo === nvcc -ccbin probe (what host compiler does it pick by default?) ===
nvcc --dryrun --keep-dir _nvcc_probe -c "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\cuda.h" 2>&1 | findstr /i "cl.exe\|HostX64\|MSVC"
