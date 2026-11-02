# Live link: Pi to PC over Wi-Fi or Tailscale

The Pi connects OUT to the PC on TCP port 5600 and streams both camera JPEGs plus a health report once per second. The PC shows the live depth view. Verified so far: protocol, receiver and sender on this PC over loopback with synthetic frames (6 automated tests pass). NOT verified: real cameras on the Pi, the Wi-Fi link, Tailscale, the Windows firewall step.

## 0. One-time PC setup
1. Find the PC's IP: PowerShell `ipconfig`, use the "IPv4 Address" of your Wi-Fi adapter (for example 192.168.1.20).
2. Allow the port through Windows Firewall (needs an Administrator PowerShell; I did not run this):
```powershell
netsh advfirewall firewall add rule name="Gabai receiver" dir=in action=allow protocol=TCP localport=5600 profile=private
```
Your Wi-Fi must be set to a Private network for that profile to apply.

## 1. One-time Pi setup
From the PC, in the project folder:
```powershell
scp -r gabai pi@<PI_IP>:~/
```
On the Pi (`ssh pi@<PI_IP>`):
```bash
sudo apt install -y python3-opencv python3-numpy v4l-utils
v4l2-ctl --list-devices        # note the two camera device paths, normally /dev/video0 and /dev/video2
```

## 2. Test the link without cameras (do this first)
PC, in the project folder:
```powershell
py -m gabai.pc.receiver --port 5600
```
Pi:
```bash
cd ~ && python3 -m gabai.pi.sender --host <PC_IP> --test-pattern
```
Expected: Pi prints `connected to <PC_IP>:5600`. PC prints one line per second like `connected=True frames=... fps=15.0 2.4 Mbit/s rejected=0 pi_temp=48.0 throttled=0x0`.
Failure signs: Pi prints `link down (... timed out)` = wrong IP or the firewall rule is missing; `rejected=1` on the PC = token mismatch (use `--token` on both sides).

## 3. Real cameras, with the depth view
Stop the receiver (Ctrl+C), then PC:
The live view uses the 640x480 the Pi sends; `--width` does not apply to it.
```powershell
py -m gabai.spike.demo_viewer --source live --model m
```
Pi:
```bash
python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video0 --right /dev/video2
```
Expected:
- The window shows the left camera, the depth map, the formula with numbers, and a LINK line: incoming FPS, Pi FPS, Pi temperature, `throttled=0x0` (anything else means power trouble), and capture-to-PC latency.
- For the first ~10 seconds it says "calibrating row alignment n/12". Slowly move the rig around a textured scene (shelves, posters). Plain walls will not work.
- If the depth looks flat or wrong, swap `--left` and `--right`.

## 4. Over the internet with Tailscale
1. PC: install Tailscale from https://tailscale.com/download and sign in. Pi: `curl -fsSL https://tailscale.com/install.sh | sh` then `sudo tailscale up` and sign in with the same account.
2. PC address: `tailscale ip -4` (looks like 100.x.y.z). Use that as `--host` on the Pi. Nothing else changes.
Notes: Tailscale encrypts the traffic; on plain Wi-Fi the video is NOT encrypted and the token is only a basic guard. Pi 3B Wi-Fi is 2.4 GHz and slower than Ethernet; expect lower FPS if the signal is weak.

## What to report
```
Step 2: connected? fps=__ Mbit/s=__ rejected=__
Step 3: Pi FPS=__ incoming FPS=__ latency=__ ms throttled=__ temp=__ depth looks (ok/flat/wrong) 
Step 4 (Tailscale): works? fps=__ latency=__ ms
Errors: paste the Pi and PC console text.
```
Stop: Ctrl+C on the Pi and PC. Restart by rerunning the same commands; the Pi reconnects by itself if the PC restarts.
