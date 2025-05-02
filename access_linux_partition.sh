#!/bin/bash

echo "Linux Partition Access Script"
echo "----------------------------"
echo "This script will create a Docker container to access the Linux partition."
echo ""

# Create a mount point
MOUNT_DIR="/tmp/linux_mount"
mkdir -p $MOUNT_DIR

echo "Starting Docker container with access to disk4s2..."
docker run --rm -it \
  --privileged \
  --device=/dev/disk4s2:/dev/sdx \
  -v $MOUNT_DIR:/mnt \
  -e DISK_DEVICE=/dev/sdx \
  ubuntu:latest \
  bash -c '
    # Install necessary tools
    apt-get update -q
    apt-get install -y e2fsprogs fdisk lsblk util-linux -q
    
    echo ""
    echo "Disk information:"
    fdisk -l $DISK_DEVICE
    
    echo ""
    echo "Attempting to mount the partition..."
    # Try to mount, including LUKS detection
    if cryptsetup isLuks $DISK_DEVICE 2>/dev/null; then
      echo "LUKS encrypted partition detected!"
      apt-get install -y cryptsetup -q
      echo "This partition requires a password to unlock."
      echo "To unlock: run \"cryptsetup open $DISK_DEVICE decrypted\" in the container"
      echo "Then mount with: \"mount /dev/mapper/decrypted /mnt\""
      bash
    else
      # Try regular mount
      mount $DISK_DEVICE /mnt
      
      if [ $? -eq 0 ]; then
        echo "Mount successful!"
        echo ""
        echo "Disk usage:"
        df -h /mnt
        
        echo ""
        echo "Directory listing:"
        ls -la /mnt
        
        echo ""
        echo "You are now in a shell with the Linux partition mounted at /mnt"
        echo "Type 'exit' when done to exit the container."
      else
        echo "Mount failed. This could be due to:"
        echo "1. Filesystem corruption"
        echo "2. Encryption (not detected as LUKS)"
        echo "3. Unsupported filesystem"
        echo ""
        echo "You can try manual mounting in the container."
      fi
      
      bash
    fi
  '

echo ""
echo "Docker container exited." 