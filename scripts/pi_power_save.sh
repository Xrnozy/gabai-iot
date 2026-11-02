#!/bin/sh
# Lowers a Raspberry Pi 3B's power draw for use on a power bank. Run on the Pi:
#     sh scripts/pi_power_save.sh        (CPU capped at 800 MHz)
#     sh scripts/pi_power_save.sh 600    (cap at 600 MHz)
# Nothing is saved: a reboot undoes all of it. It does not touch the cameras or the I2C sensors. It turns Wi-Fi power saving OFF (it makes frames arrive in bursts with stalls).
MAX_MHZ="${1:-800}"

echo "before: $(vcgencmd get_throttled)   $(vcgencmd measure_temp)   cpu $(($(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000)) MHz"

# HDMI output off (about 25 mA)
vcgencmd display_power 0 >/dev/null 2>&1 || tvservice -o >/dev/null 2>&1

# Bluetooth off
sudo rfkill block bluetooth 2>/dev/null
sudo systemctl stop bluetooth hciuart 2>/dev/null

# the green activity LED off
for led in /sys/class/leds/ACT /sys/class/leds/led0; do
    [ -d "$led" ] && echo none | sudo tee "$led/trigger" >/dev/null 2>&1 && echo 0 | sudo tee "$led/brightness" >/dev/null 2>&1
done

# Wi-Fi power saving off: no stalls of hundreds of ms while the radio sleeps
sudo iw dev wlan0 set power_save off 2>/dev/null && echo "Wi-Fi power_save: $(iw dev wlan0 get power_save 2>/dev/null)"

# cap the CPU clock: lower current peaks, which is what pulls a weak supply under 4.63 V
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq; do
    echo $((MAX_MHZ * 1000)) | sudo tee "$f" >/dev/null 2>&1
done

echo "applied: HDMI off, Bluetooth off, activity LED off, CPU max $MAX_MHZ MHz"
echo "now start the sender with --low-power, wait 30 seconds, then run:  vcgencmd get_throttled"
echo "(bit 0x1 = under-voltage RIGHT NOW; 0x10000 = it happened since boot and only a reboot clears it)"
