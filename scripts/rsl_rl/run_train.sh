#!/bin/bash
# 檢查是否已有虛擬顯示器運行
if ! pgrep -f "Xvfb :99" > /dev/null; then
    echo "Starting virtual display..."
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
    sleep 3
fi

# 設置環境變量
export DISPLAY=:99
export OMNI_RTX_FALLBACK_ADAPTER=cuda
export OMNI_KIT_ACCEPT_EULA=YES

# 執行訓練
python train.py --task=Template-Height-Scan-Go2-v0 --max_iterations=10000 --headless

# 清理（可選）
# pkill -f "Xvfb :99"
