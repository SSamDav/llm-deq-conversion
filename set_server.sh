#!/bin/bash

# Create tmux configuration
tmux_config="$(cat <<-EOF
set -g default-terminal "tmux-256color"
setw -g mode-keys vi
EOF
)"
echo "$tmux_config" > ~/.tmux.conf

# Download Helix editor
wget -O ~/hx.tar.xz https://github.com/helix-editor/helix/releases/download/25.01.1/helix-25.01.1-x86_64-linux.tar.xz

# Create hx directory if it doesn't exist
mkdir -p ~/hx

# Extract Helix editor
tar -xf ~/hx.tar.xz -C ~/hx

# Update PATH in .bashrc if not already present
if ! grep -q 'export PATH=\$PATH:$HOME/hx' ~/.bashrc; then
    echo "export PATH=\$PATH:\$HOME/hx" >> ~/.bashrc
fi

# Create Helix config file
file="$HOME/.config/helix/config.toml" 
mkdir -p "$(dirname "$file")" && touch "$file"

# Create Helix configuration
helix_config="$(cat <<-EOF
theme = "github_dark"

[editor]
end-of-line-diagnostics = "hint"

[editor.inline-diagnostics]
cursor-line = "error" # Show inline diagnostics when the cursor is on the line
other-lines = "disable" # Don't expand diagnostics unless the cursor is on the line

[editor.soft-wrap]
enable = true
max-wrap = 25 # increase value to reduce forced mid-word wrapping
max-indent-retain = 0
wrap-indicator = "" # set wrap-indicator to "" to hide it
EOF
)"
echo "$helix_config" > "$file"
