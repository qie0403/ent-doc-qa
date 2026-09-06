@echo off
chcp 65001 >nul
echo ============================================
echo   企业级文档知识库问答系统 - 启动脚本
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
echo Python 环境正常
echo.

echo [2/3] 检查 .env 配置文件...
if not exist ".env" (
    echo [警告] 未找到 .env 文件，正在从 .env.example 复制...
    copy ".env.example" ".env" >nul
    echo 请编辑 .env 文件填入 API Key 后重新运行
    notepad ".env"
    pause
    exit /b 0
)
echo .env 配置文件存在
echo.

echo [3/3] 启动 Streamlit 应用...
echo.
echo 浏览器将自动打开 http://localhost:8501
echo 按 Ctrl+C 可停止服务
echo.
echo ============================================
echo.

streamlit run app/main.py

pause
