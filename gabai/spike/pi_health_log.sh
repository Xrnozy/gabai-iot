#!/bin/bash
# Runs on the Pi alongside pi_capture.py. Logs power/thermal/memory once per second.
# throttled=0x0 is healthy. Bit 0 = under-voltage now, bit 16 = under-voltage has occurred.
OUT=${1:-health.csv}
echo "time,throttled,temp,mem_available_mb,load1" > "$OUT"
while true; do
  T=$(vcgencmd get_throttled | cut -d= -f2)
  C=$(vcgencmd measure_temp | cut -d= -f2)
  M=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  L=$(cut -d' ' -f1 /proc/loadavg)
  echo "$(date +%s),$T,$C,$M,$L" >> "$OUT"
  sleep 1
done
