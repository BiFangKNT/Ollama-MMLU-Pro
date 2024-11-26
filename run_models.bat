@echo off
chcp 65001 > nul

:: 请求管理员权限
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo 请求管理员权限...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
    pushd "%CD%"
    CD /D "%~dp0"

:: 设置完整工作路径
set "WORK_DIR=%~dp0"
cd /d "%WORK_DIR%"
set CUDA_VISIBLE_DEVICES=0
set OMP_NUM_THREADS=12
set OLLAMA_NUM_PARALLEL=4

:: 激活conda环境
call conda activate base

:: 定义待测试模型列表
set models=qwen2.5:32b-instruct-q2_K

:: qwen2.5:7b glm4:9b-chat-q4_K_M qwen2.5:14b-instruct-q3_K_M deepseek-v2:16b-lite-chat-q3_K_M 


:: 遍历每个模型并执行测试
for %%m in (%models%) do (
    echo.
    echo 正在测试模型: %%m
    powershell -Command "(Get-Content config.toml) -replace 'model = .*', 'model = \"%%m\"' | Set-Content config.toml"
    
    :: 运行Python脚本并捕获错误
    call conda activate base && python run_ollama.py
    if errorlevel 1 (
        echo 模型 %%m 测试过程中出现错误
    ) else (
        echo 模型 %%m 测试完成
    )
    
    echo.
    choice /C YN /T 5 /D Y /M "是否继续下一个测试？5秒后自动继续(Y=是，N=否)"
    if errorlevel 2 (
        echo 测试已手动终止
        goto :eof
    )
    timeout /t 5 > nul
)

echo 所有测试已完成
pause