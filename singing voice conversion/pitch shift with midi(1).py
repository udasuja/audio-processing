import os
import numpy as np
import librosa
import pretty_midi
import pyworld as pw
import soundfile as sf
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import librosa.sequence
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --------------------------------------------------
# 0. 경로 설정
# --------------------------------------------------
midi_path = "/content/작은별.mid"
speech_path = "/content/작은별 발음 정확.wav"
output_filename = "/content/song_output_final.wav"

# --------------------------------------------------
# 1. 공통 설정
# --------------------------------------------------
sr = 44100
hop_length = 220  #MIDI 시간을 frame 단위로 바꾸기 위해서 사용된다.
frame_period = 5  # ms 단위 (pitch shift를 하기 위해 frame단위로 다시 쪼개는데, 그때의 길이)
trim_db = 25  # librosa trim 기준

# 파일 존재 체크
if not os.path.exists(midi_path):
    raise FileNotFoundError(f"MIDI 파일을 찾을 수 없습니다: {midi_path}")
if not os.path.exists(speech_path):
    raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {speech_path}")

#저역 통과 필터
def lowpass_filter(signal, cutoff, sr, order=6):
    nyquist = sr / 2
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered
def formant_shift_spectrum(sp, ap, sr, ratio):
    """
    sp, ap: (n_frames, n_bins)
    ratio: formant shift ratio (1.0 = no shift, 1.1 = 10% brighter, 0.9 = darker)
    """
    n_frames, n_bins = sp.shape
    orig_freqs = np.linspace(0, sr/2, n_bins)

    warped_sp = np.zeros_like(sp)
    warped_ap = np.zeros_like(ap)

    for i in range(n_frames):
        sp_frame = sp[i]
        ap_frame = ap[i]

        # frequency warping
        new_freqs = orig_freqs / ratio
        new_freqs = np.clip(new_freqs, 0, sr/2)

        f_sp = interp1d(new_freqs, sp_frame, kind='linear',
                        bounds_error=False, fill_value=(sp_frame[0], sp_frame[-1]))
        f_ap = interp1d(new_freqs, ap_frame, kind='linear',
                        bounds_error=False, fill_value=(ap_frame[0], ap_frame[-1]))

        warped_sp[i] = f_sp(orig_freqs)
        warped_ap[i] = f_ap(orig_freqs)

    return warped_sp, warped_ap

# --------------------------------------------------
# 2. MIDI -> frame 단위 F0(Hz) 추출
# --------------------------------------------------
print("\n[1단계] MIDI 분석")

pm = pretty_midi.PrettyMIDI(midi_path)
if len(pm.instruments) == 0:
    raise ValueError("MIDI에 instrument 트랙이 없습니다.")

instrument = pm.instruments[0]
total_duration = pm.get_end_time()

# hop_length 기준 프레임 개수
n_frames_midi = int(np.ceil(total_duration * sr / hop_length))
target_f0 = np.zeros(n_frames_midi, dtype=np.float64)

for note in instrument.notes:
    start_frame = int(np.round(note.start * sr / hop_length))
    end_frame = int(np.round(note.end * sr / hop_length))
    start_frame = max(0, start_frame)
    end_frame = min(n_frames_midi, end_frame)
    if start_frame >= end_frame:
        continue
    pitch_hz = pretty_midi.note_number_to_hz(note.pitch)
    target_f0[start_frame:end_frame] = pitch_hz

# 앞쪽 노트 없는 묵음 프레임 제거
nonzero_idx = np.where(target_f0 > 0)[0]
if len(nonzero_idx) == 0:
    raise ValueError("MIDI에서 F0>0인 노트를 찾을 수 없습니다.")

first_note_idx = nonzero_idx[0]
target_f0_trimmed = target_f0[first_note_idx:]

print(f"  MIDI F0 프레임 수: 원본 {len(target_f0)} -> trimmed {len(target_f0_trimmed)}")

# --------------------------------------------------
# 3. 음성 -> WORLD(F0, SP, AP) 분석
# --------------------------------------------------
print("\n[2단계] 음성 분석")

# 원본 SR로 로드 후, 44100으로 리샘플
x, sr_orig = librosa.load(speech_path, sr=None)
x_resampled = librosa.resample(x, orig_sr=sr_orig, target_sr=sr)

# 앞뒤 묵음 제거
x_trimmed, _ = librosa.effects.trim(x_resampled, top_db=trim_db)
print(f"  음성 길이: 원본 {len(x_resampled)/sr:.2f}s -> trimmed {len(x_trimmed)/sr:.2f}s")

x64 = x_trimmed.astype(np.float64)

# WORLD 분석
_f0, t = pw.dio(x64, sr, frame_period=frame_period)
f0 = pw.stonemask(x64, _f0, t, sr)
sp = pw.cheaptrick(x64, f0, t, sr)
ap = pw.d4c(x64, f0, t, sr)

# 앞부분 무성(F0==0) 제거
voiced_idx = np.where(f0 > 0)[0]
if len(voiced_idx) == 0:
    raise ValueError("음성에서 유성 구간(F0>0)을 찾지 못했습니다.")

first_voice_idx = voiced_idx[0]
f0_trimmed = f0[first_voice_idx:]
sp_trimmed = sp[first_voice_idx:]
ap_trimmed = ap[first_voice_idx:]

print(f"  WORLD 프레임 수: 원본 {len(f0)} -> trimmed {len(f0_trimmed)}")

# --------------------------------------------------
# 4. DTW (기준을 MIDI로 설정)
#     - MIDI를 기준 축(axis0)
#     - 음성을 비교 축(axis1)
# --------------------------------------------------
print("\n[3단계] DTW 정렬 (v/uv 기반, 기준 = MIDI)")

# MIDI = 기준(reference), 음성 = 비교(sequence)
ref_voiced = (target_f0_trimmed > 0)   # 기준축
qry_voiced = (f0_trimmed > 0)          # 비교축

# 비용행렬: (기준축, 비교축)
# 기존과 반대가 됨
cost_matrix = (ref_voiced[:, None] != qry_voiced[None, :]).astype(np.float64)

D, wp = librosa.sequence.dtw(C=cost_matrix)
wp = np.flip(wp, axis=0)

# 이제 wp[:,0] = MIDI 프레임, wp[:,1] = 음성 프레임
midi_idx = wp[:, 0]
speech_idx = wp[:, 1]

# 인덱스 범위 보정
midi_idx = np.clip(midi_idx, 0, len(target_f0_trimmed) - 1)
speech_idx = np.clip(speech_idx, 0, len(f0_trimmed) - 1)

print(f"  DTW path length: {len(wp)}")
print(f"  MIDI frames: {len(target_f0_trimmed)}, Speech frames: {len(f0_trimmed)}")

# --------------------------------------------------
# 5. 방식 B 적용:
#    - SP/AP는 음성에서 가져와서 DTW로 시간만 warp (formant 구조 유지)
#    - F0는 MIDI에서 가져와서 DTW로 warp (pitch 완전 교체)
# --------------------------------------------------
print("\n[4단계] SP/AP/F0 warp 및 방식 B 적용")

L_midi = len(target_f0_trimmed)
midi_to_speech = np.zeros(L_midi, dtype=int)

for i in range(L_midi):
    # 현재 MIDI 프레임 i와 가장 가까운 DTW path 지점 찾기
    nearest = np.argmin(np.abs(midi_idx - i))
    midi_to_speech[i] = speech_idx[nearest]

# ------------------------------
# 🔥 최종 warp 결과는 MIDI 길이에 정확히 맞춤
# ------------------------------
warped_sp = sp_trimmed[midi_to_speech]
warped_ap = ap_trimmed[midi_to_speech]
warped_f0_midi = target_f0_trimmed   # 이미 길이 = MIDI 길이

print(f"  warped_sp shape: {warped_sp.shape}")
print(f"  warped_ap shape: {warped_ap.shape}")
print(f"  warped_f0_midi length: {len(warped_f0_midi)}")

formant_ratio = 0.8
warped_sp, warped_ap = formant_shift_spectrum(
    warped_sp, warped_ap, sr, formant_ratio
)
print(f" - 스펙트럼/F0 정렬 완료.")

# --------------------------------------------------
# 6. WORLD 합성
#    - 합성에 들어가는 F0는 warped_f0_midi = MIDI 기반
#    - 즉, 이론적으로 모든 frame의 pitch는 MIDI와 동일
# --------------------------------------------------
print("\n[5단계] WORLD 합성 (F0 = MIDI)")

synthesized = pw.synthesize(
    warped_f0_midi,  # MIDI에서 온 F0 (DTW warp된 상태)
    warped_sp,       # 음성에서 온 formant 구조
    warped_ap,       # 음성에서 온 aperiodicity
    sr,
    frame_period=frame_period
)


# ====== [추가: LPF 적용] =======
cutoff_hz = 5000
synthesized = lowpass_filter(synthesized, cutoff_hz, sr)

sf.write(output_filename, synthesized, sr)
print(f"  합성 완료: {output_filename}")