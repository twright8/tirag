#!/bin/bash
# Script to increase inotify watch limits

echo "Current inotify watch limits:"
echo "fs.inotify.max_user_watches = $(cat /proc/sys/fs/inotify/max_user_watches)"
echo "fs.inotify.max_user_instances = $(cat /proc/sys/fs/inotify/max_user_instances)"

echo -e "\nTemporarily increasing limits for the current session..."
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=512

echo -e "\nNew inotify watch limits:"
echo "fs.inotify.max_user_watches = $(cat /proc/sys/fs/inotify/max_user_watches)"
echo "fs.inotify.max_user_instances = $(cat /proc/sys/fs/inotify/max_user_instances)"

echo -e "\nTo make these changes permanent, run the following command:"
echo "echo 'fs.inotify.max_user_watches=524288' | sudo tee -a /etc/sysctl.conf"
echo "echo 'fs.inotify.max_user_instances=512' | sudo tee -a /etc/sysctl.conf"
echo "sudo sysctl -p"

echo -e "\nScript complete. The changes are temporary and will be reset on system reboot."
