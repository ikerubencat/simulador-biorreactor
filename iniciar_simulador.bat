@echo off
chcp 65001 >nul
title Simulador de Biorreactores 3D - Servidor Local
cd /d "%~dp0"

echo ========================================================
echo   Iniciando Simulador Computacional de Biorreactores 3D
echo   No cierres esta ventana mientras uses la aplicacion.
echo ========================================================
echo.

.\.venv\Scripts\streamlit.exe run app.py --server.port 8501
pause
