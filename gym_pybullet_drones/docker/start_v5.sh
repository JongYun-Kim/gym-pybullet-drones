#!/bin/bash
# 백그라운드에서 SSH 데몬 실행 (포그라운드 실행 옵션 -D 없이 &로 실행)
# 포트는 Dockerfile에서 2222로 설정해두었음
echo "(start.sh) Starting SSH daemon... in background"
/usr/sbin/sshd &

# Python 스크립트를 실행 (완료될 때까지 대기)
echo "(start.sh) Running Python script..."
python3 /home/ray/resources/test_torch.py

# 기본적으로 bash 쉘을 실행하여 인터랙티브 개발 환경 제공
echo "(start.sh) Starting bash shell..."
exec bash