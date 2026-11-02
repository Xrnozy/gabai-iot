"""Render the fixed alert vocabulary to WAV files with Windows' built-in offline voice (no download, no internet).

    py -m gabai.tools.make_alert_audio [--voice "Microsoft Zira Desktop"] [--rate 1]

Writes gabai/audio/alerts/<clip>.wav (23 clips, about 1 second each). Rerun after changing a phrase in gabai/speaker.py.
Piper or any other engine would also work: just produce WAV files with the same names.
"""
import argparse
import json
import os
import subprocess
import tempfile
import wave

import numpy as np

from gabai.speaker import all_clips, phrase_for

PS = r"""
param([string]$spec, [string]$outdir, [string]$voice, [int]$rate)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $s.SelectVoice($voice) } catch { Write-Host "voice '$voice' not found, using the default" }
$s.Rate = $rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$items = Get-Content -Raw $spec | ConvertFrom-Json
foreach ($p in $items.PSObject.Properties) {
    $path = Join-Path $outdir ($p.Name + ".wav")
    $s.SetOutputToWaveFile($path, $fmt)
    $s.Speak($p.Value)
    $s.SetOutputToNull()
}
$s.Dispose()
"""


def trim(path, threshold=250, pad_s=0.03):
    """Cut the silence the voice engine leaves at both ends, so an alert starts speaking right away."""
    with wave.open(path) as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    x = np.frombuffer(frames, np.int16)
    loud = np.nonzero(np.abs(x) > threshold)[0]
    if len(loud) == 0:
        return
    pad = int(pad_s * params.framerate)
    y = x[max(loud[0] - pad, 0): loud[-1] + pad]
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(y.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice", default="Microsoft Zira Desktop")
    ap.add_argument("--rate", type=int, default=3, help="-10 slow .. 10 fast")
    ap.add_argument("--out", default=os.path.join("gabai", "audio", "alerts"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = {c: phrase_for(c) for c in all_clips()}
    with tempfile.TemporaryDirectory() as d:
        sp, ps = os.path.join(d, "spec.json"), os.path.join(d, "render.ps1")
        json.dump(spec, open(sp, "w"))
        open(ps, "w").write(PS)
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps, sp,
                            os.path.abspath(a.out), a.voice, str(a.rate)], capture_output=True, text=True)
    print(r.stdout.strip() or "", r.stderr.strip() or "")
    for c in all_clips():
        if os.path.exists(os.path.join(a.out, c + ".wav")):
            trim(os.path.join(a.out, c + ".wav"))
    done = [c for c in all_clips() if os.path.exists(os.path.join(a.out, c + ".wav"))]
    print(f"{len(done)}/{len(spec)} clips in {a.out}")


if __name__ == "__main__":
    main()
