#!/bin/bash
tmux_config="$(cat <<-EOF
set -g default-terminal "tmux-256color"
setw -g mode-keys vi
EOF
)"
echo "$tmux_config" > ~/.tmux.conf

wget -O ~/hx.tar.xz https://github.com/helix-editor/helix/releases/download/25.01.1/helix-25.01.1-x86_64-linux.tar.xz
tar -xf ~/hx.tar.xz -C ~/hx

echo  "export PATH=$PATH:~/hx" >> ~/.bashrc 
