//! Feature-gated transcription loader using aprender for speech-to-text.
//!
//! When a media file has a sidecar subtitle (`.srt` or `.vtt`) adjacent to it,
//! the subtitle is loaded directly without transcription. For WAV files without
//! sidecars, the audio is decoded, resampled to 16 kHz, and processed through
//! a Whisper-compatible mel spectrogram via aprender.
//!
//! Full ASR inference requires a concrete `AsrModel` implementation in aprender.
//! The mel spectrogram pipeline is ready; model integration arrives when aprender
//! ships Whisper GGUF/safetensors inference.

use crate::loader::subtitle::SubtitleLoader;
use crate::loader::{DocumentLoader, LoaderRegistry};
use crate::media::{SubtitleCue, SubtitleFormat, SubtitleTrack};
use crate::{Document, Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Media file extensions supported by the transcription loader.
const MEDIA_EXTENSIONS: &[&str] = &["mp4", "mp3", "wav", "m4a", "ogg", "flac", "webm"];

/// Compute backend for transcription inference.
#[derive(Debug, Clone, Copy, Default)]
pub enum TranscriptionBackend {
    /// CPU with SIMD acceleration via trueno
    #[default]
    Cpu,
    /// GPU via wgpu (cross-platform)
    Gpu,
    /// NVIDIA CUDA (Linux/Windows)
    Cuda,
}

/// Configuration for the transcription pipeline.
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    /// Language hint (ISO 639-1, e.g., "en"). `None` for auto-detect.
    pub language: Option<String>,
    /// Beam size for decoding (1 = greedy, 5 = default).
    pub beam_size: usize,
    /// Enable word-level timestamps (more precise but slower).
    pub word_timestamps: bool,
    /// Write `.srt` sidecar files after transcription for caching.
    pub write_sidecar: bool,
    /// Compute backend for inference.
    pub backend: TranscriptionBackend,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            language: Some("en".into()),
            beam_size: 5,
            word_timestamps: false,
            write_sidecar: true,
            backend: TranscriptionBackend::default(),
        }
    }
}

/// Loader that handles media files via sidecar subtitle detection
/// and aprender-based speech-to-text transcription.
///
/// When a media file has a sidecar subtitle (`.srt` or `.vtt`) adjacent to it,
/// the subtitle is loaded directly. Otherwise, the audio pipeline computes a
/// Whisper-compatible mel spectrogram for ASR inference.
///
/// # Example
///
/// ```rust,no_run
/// use trueno_rag::loader::transcription::{TranscriptionLoader, TranscriptionConfig};
/// use trueno_rag::loader::LoaderRegistry;
///
/// let mut registry = LoaderRegistry::new();
/// registry.register(Box::new(TranscriptionLoader::with_defaults()));
/// // Now the registry handles .mp4, .wav, etc. via sidecar detection
/// ```
pub struct TranscriptionLoader {
    config: TranscriptionConfig,
    mel_filterbank: aprender::audio::MelFilterbank,
}

impl TranscriptionLoader {
    /// Create a new transcription loader with the given configuration.
    #[must_use]
    pub fn new(config: TranscriptionConfig) -> Self {
        let mel_config = aprender::audio::MelConfig::whisper();
        let mel_filterbank = aprender::audio::MelFilterbank::new(&mel_config);
        Self {
            config,
            mel_filterbank,
        }
    }

    /// Create a loader with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TranscriptionConfig::default())
    }

    /// Compute mel spectrogram from audio samples.
    ///
    /// Resamples to 16 kHz if needed, then computes an 80-bin mel spectrogram
    /// using Whisper-compatible parameters (400-pt FFT, 160 hop length).
    pub fn compute_mel(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let samples_16k = if sample_rate == 16000 {
            samples.to_vec()
        } else {
            aprender::audio::resample(samples, sample_rate, 16000)
                .map_err(|e| Error::InvalidInput(format!("Resample failed: {e}")))?
        };

        self.mel_filterbank
            .compute(&samples_16k)
            .map_err(|e| Error::InvalidInput(format!("Mel computation failed: {e}")))
    }

    /// Access the transcription configuration.
    #[must_use]
    pub fn config(&self) -> &TranscriptionConfig {
        &self.config
    }
}

impl DocumentLoader for TranscriptionLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        MEDIA_EXTENSIONS.to_vec()
    }

    fn load(&self, path: &Path) -> Result<Document> {
        // 1. Check for sidecar subtitle file
        if let Some(sidecar) = LoaderRegistry::find_sidecar(path) {
            return SubtitleLoader.load(&sidecar);
        }

        // 2. Read audio (WAV support for now)
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        if ext != "wav" {
            return Err(Error::InvalidInput(format!(
                "Direct transcription for .{ext} files requires audio codec support. \
                 Provide a .srt or .vtt sidecar file alongside the media, \
                 or convert to WAV first: {}",
                path.display()
            )));
        }

        let (samples, sample_rate, channels) = read_wav(path)?;

        // 3. Convert to mono
        let mono = if channels > 1 {
            stereo_to_mono(&samples, channels)
        } else {
            samples
        };

        // 4. Compute mel spectrogram (Whisper-compatible: 80 mels, 16 kHz)
        let mel = self.compute_mel(&mono, sample_rate)?;
        let n_frames = mel.len() / 80;

        // 5. ASR inference requires a model — not yet available
        // The mel spectrogram is computed and ready; actual inference
        // requires a concrete AsrModel implementation in aprender.
        Err(Error::InvalidInput(format!(
            "Transcription model not configured. Mel spectrogram computed \
             ({n_frames} frames) but ASR inference requires a Whisper model. \
             Provide a .srt sidecar file alongside: {}",
            path.display()
        )))
    }
}

impl std::fmt::Debug for TranscriptionLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptionLoader")
            .field("config", &self.config)
            .field("mel_bins", &80)
            .finish_non_exhaustive()
    }
}

/// Convert aprender ASR segments to a [`SubtitleTrack`].
///
/// Maps millisecond timestamps from aprender's `Segment` type to
/// the fractional-seconds representation used by `SubtitleCue`.
#[allow(clippy::cast_precision_loss)]
pub fn segments_to_track(segments: &[aprender::speech::Segment]) -> SubtitleTrack {
    let cues = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| SubtitleCue {
            index: i,
            start_secs: seg.start_ms as f64 / 1000.0,
            end_secs: seg.end_ms as f64 / 1000.0,
            text: seg.text.trim().to_string(),
        })
        .collect();
    SubtitleTrack {
        format: SubtitleFormat::Srt,
        cues,
    }
}

/// Build a [`Document`] from a transcription result.
pub fn build_transcription_document(path: &Path, track: &SubtitleTrack) -> Result<Document> {
    let title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled")
        .to_string();

    let mut metadata = HashMap::new();
    metadata.insert(
        "duration_secs".into(),
        serde_json::json!(track.duration_secs()),
    );
    metadata.insert("format".into(), serde_json::json!("transcription"));
    metadata.insert("cue_count".into(), serde_json::json!(track.cues.len()));
    metadata.insert(
        "subtitle_cues".into(),
        serde_json::to_value(&track.cues).map_err(Error::Serialization)?,
    );

    let mut doc = Document::new(track.to_plain_text())
        .with_title(title)
        .with_source(path.to_string_lossy());
    doc.metadata = metadata;
    Ok(doc)
}

/// Write a [`SubtitleTrack`] as an SRT sidecar file adjacent to a media file.
///
/// Returns the path of the written sidecar.
pub fn write_sidecar(media_path: &Path, track: &SubtitleTrack) -> Result<std::path::PathBuf> {
    let sidecar_path = media_path.with_extension("srt");
    let srt_content = track.to_srt_string();
    std::fs::write(&sidecar_path, srt_content).map_err(Error::Io)?;
    Ok(sidecar_path)
}

// ── Minimal WAV parser ──────────────────────────────────────────────────

/// Read a WAV file into PCM f32 samples, returning (samples, sample_rate, channels).
fn read_wav(path: &Path) -> Result<(Vec<f32>, u32, u8)> {
    let data = std::fs::read(path).map_err(Error::Io)?;
    parse_wav(&data)
}

/// WAV format metadata parsed from the fmt chunk.
struct WavFmt {
    audio_format: u16,
    channels: u8,
    sample_rate: u32,
    bits_per_sample: u16,
}

/// Parse the fmt chunk of a WAV file.
fn parse_wav_fmt(fmt_data: &[u8]) -> WavFmt {
    WavFmt {
        audio_format: u16::from_le_bytes([fmt_data[0], fmt_data[1]]),
        channels: u16::from_le_bytes([fmt_data[2], fmt_data[3]]) as u8,
        sample_rate: u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]),
        bits_per_sample: u16::from_le_bytes([fmt_data[14], fmt_data[15]]),
    }
}

/// Parse a WAV byte buffer into PCM f32 samples.
///
/// Supports PCM 16-bit, PCM 24-bit, and IEEE float 32-bit formats.
fn parse_wav(data: &[u8]) -> Result<(Vec<f32>, u32, u8)> {
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(Error::InvalidInput("Not a valid WAV file".into()));
    }

    let mut pos = 12;
    let mut fmt = WavFmt { audio_format: 0, channels: 0, sample_rate: 0, bits_per_sample: 0 };

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;

        if chunk_id == b"fmt " && chunk_size >= 16 && pos + 8 + chunk_size <= data.len() {
            fmt = parse_wav_fmt(&data[pos + 8..]);
        } else if chunk_id == b"data" {
            let data_end = (pos + 8 + chunk_size).min(data.len());
            let samples = decode_wav_samples(&data[pos + 8..data_end], fmt.audio_format, fmt.bits_per_sample)?;
            return Ok((samples, fmt.sample_rate, fmt.channels));
        }

        pos += 8 + chunk_size;
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }

    Err(Error::InvalidInput("WAV file missing data chunk".into()))
}

/// Decode raw WAV sample bytes to f32 based on format and bit depth.
fn decode_wav_samples(data: &[u8], audio_format: u16, bits_per_sample: u16) -> Result<Vec<f32>> {
    match (audio_format, bits_per_sample) {
        // PCM 16-bit
        (1, 16) => Ok(data
            .chunks_exact(2)
            .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
            .collect()),
        // PCM 24-bit (sign-extend via arithmetic shift)
        (1, 24) => Ok(data
            .chunks_exact(3)
            .map(|c| {
                let val = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                val as f32 / 8_388_608.0
            })
            .collect()),
        // IEEE float 32-bit
        (3, 32) => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        _ => Err(Error::InvalidInput(format!(
            "Unsupported WAV format: format={audio_format}, bits={bits_per_sample}"
        ))),
    }
}

/// Convert interleaved multi-channel audio to mono by averaging channels.
fn stereo_to_mono(samples: &[f32], channels: u8) -> Vec<f32> {
    let ch = channels as usize;
    if ch <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_config_default() {
        let config = TranscriptionConfig::default();
        assert_eq!(config.language, Some("en".into()));
        assert_eq!(config.beam_size, 5);
        assert!(!config.word_timestamps);
        assert!(config.write_sidecar);
    }

    #[test]
    fn test_transcription_backend_default() {
        let backend = TranscriptionBackend::default();
        assert!(matches!(backend, TranscriptionBackend::Cpu));
    }

    #[test]
    fn test_media_extensions() {
        let loader = TranscriptionLoader::with_defaults();
        let exts = loader.supported_extensions();
        assert!(exts.contains(&"mp4"));
        assert!(exts.contains(&"wav"));
        assert!(exts.contains(&"mp3"));
        assert!(exts.contains(&"flac"));
        assert!(exts.contains(&"webm"));
    }

    #[test]
    fn test_segments_to_track() {
        let segments = vec![
            aprender::speech::Segment::new("Hello world.", 0, 3000),
            aprender::speech::Segment::new("How are you?", 3500, 6000),
        ];
        let track = segments_to_track(&segments);
        assert_eq!(track.cues.len(), 2);
        assert_eq!(track.cues[0].text, "Hello world.");
        assert!((track.cues[0].start_secs).abs() < 0.001);
        assert!((track.cues[0].end_secs - 3.0).abs() < 0.001);
        assert!((track.cues[1].start_secs - 3.5).abs() < 0.001);
        assert!((track.cues[1].end_secs - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_segments_to_track_empty() {
        let track = segments_to_track(&[]);
        assert!(track.cues.is_empty());
        assert!((track.duration_secs()).abs() < 0.001);
    }

    #[test]
    fn test_parse_wav_invalid() {
        assert!(parse_wav(b"not a wav file").is_err());
        assert!(parse_wav(b"").is_err());
    }

    #[test]
    fn test_parse_wav_too_short() {
        assert!(parse_wav(b"RIFF").is_err());
    }

    #[test]
    fn test_parse_wav_pcm16() {
        let wav = build_test_wav_pcm16(&[0, 16384, -16384, 32767], 16000, 1);
        let (samples, rate, channels) = parse_wav(&wav).unwrap();
        assert_eq!(rate, 16000);
        assert_eq!(channels, 1);
        assert_eq!(samples.len(), 4);
        assert!((samples[0]).abs() < 0.001);
        assert!((samples[1] - 0.5).abs() < 0.01);
        assert!((samples[2] + 0.5).abs() < 0.01);
        assert!((samples[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_wav_float32() {
        let wav = build_test_wav_float32(&[0.0, 0.5, -0.5, 1.0], 44100, 1);
        let (samples, rate, channels) = parse_wav(&wav).unwrap();
        assert_eq!(rate, 44100);
        assert_eq!(channels, 1);
        assert_eq!(samples.len(), 4);
        assert!((samples[0]).abs() < 0.001);
        assert!((samples[1] - 0.5).abs() < 0.001);
        assert!((samples[2] + 0.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_wav_missing_data_chunk() {
        // RIFF header + fmt chunk but no data chunk
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&36u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // 1 ch
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        assert!(parse_wav(&wav).is_err());
    }

    #[test]
    fn test_stereo_to_mono() {
        let stereo = vec![0.5, -0.5, 1.0, 0.0, -1.0, 1.0];
        let mono = stereo_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0]).abs() < 0.001); // (0.5 + -0.5) / 2
        assert!((mono[1] - 0.5).abs() < 0.001); // (1.0 + 0.0) / 2
        assert!((mono[2]).abs() < 0.001); // (-1.0 + 1.0) / 2
    }

    #[test]
    fn test_stereo_to_mono_passthrough() {
        let mono_input = vec![0.1, 0.2, 0.3];
        let result = stereo_to_mono(&mono_input, 1);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_load_non_wav_media_errors_helpful() {
        let loader = TranscriptionLoader::with_defaults();
        let result = loader.load(Path::new("/tmp/nonexistent_video.mp4"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("codec") || err.contains("sidecar") || err.contains("WAV"));
    }

    #[test]
    fn test_sidecar_fallback() {
        let dir = std::env::temp_dir().join("trueno_rag_test_transcription_sidecar");
        let _ = std::fs::create_dir_all(&dir);
        let media = dir.join("lecture.wav");
        let srt = dir.join("lecture.srt");
        std::fs::write(&media, b"fake wav data").unwrap();
        std::fs::write(
            &srt,
            "1\n00:00:01,000 --> 00:00:04,500\nSidecar text.\n",
        )
        .unwrap();

        let loader = TranscriptionLoader::with_defaults();
        let doc = loader.load(&media).unwrap();
        assert!(doc.content.contains("Sidecar text"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_build_transcription_document() {
        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: vec![
                SubtitleCue {
                    index: 0,
                    start_secs: 0.0,
                    end_secs: 3.0,
                    text: "Hello".into(),
                },
                SubtitleCue {
                    index: 1,
                    start_secs: 3.0,
                    end_secs: 6.0,
                    text: "World".into(),
                },
            ],
        };
        let doc = build_transcription_document(Path::new("/tmp/test.wav"), &track).unwrap();
        assert_eq!(doc.content, "Hello World");
        assert_eq!(doc.title, Some("test".into()));
        assert!(doc.metadata.contains_key("duration_secs"));
        assert!(doc.metadata.contains_key("subtitle_cues"));
        assert!(doc.metadata.contains_key("cue_count"));
    }

    #[test]
    fn test_write_sidecar() {
        let dir = std::env::temp_dir().join("trueno_rag_test_write_sidecar");
        let _ = std::fs::create_dir_all(&dir);
        let media = dir.join("output.mp4");

        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: vec![SubtitleCue {
                index: 0,
                start_secs: 1.0,
                end_secs: 4.5,
                text: "Hello.".into(),
            }],
        };

        let sidecar = write_sidecar(&media, &track).unwrap();
        assert_eq!(sidecar.extension().unwrap(), "srt");
        let content = std::fs::read_to_string(&sidecar).unwrap();
        assert!(content.contains("Hello."));
        assert!(content.contains("00:00:01,000"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mel_computation() {
        let loader = TranscriptionLoader::with_defaults();
        // 1 second of silence at 16 kHz
        let silence = vec![0.0f32; 16000];
        let mel = loader.compute_mel(&silence, 16000);
        assert!(mel.is_ok());
        let mel = mel.unwrap();
        assert!(!mel.is_empty());
    }

    #[test]
    fn test_mel_computation_resamples() {
        let loader = TranscriptionLoader::with_defaults();
        // 1 second at 44.1 kHz — should resample to 16 kHz internally
        let audio = vec![0.0f32; 44100];
        let mel = loader.compute_mel(&audio, 44100);
        assert!(mel.is_ok());
    }

    #[test]
    fn test_loader_debug() {
        let loader = TranscriptionLoader::with_defaults();
        let debug = format!("{loader:?}");
        assert!(debug.contains("TranscriptionLoader"));
        assert!(debug.contains("mel_bins"));
    }

    // ── Test helpers ─────────────────────────────────────────────

    fn build_test_wav_pcm16(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let data_size = (samples.len() * 2) as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * u32::from(channels) * 2).to_le_bytes());
        wav.extend_from_slice(&(channels * 2).to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        wav
    }

    fn build_test_wav_float32(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
        let data_size = (samples.len() * 4) as u32;
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        // fmt chunk
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * u32::from(channels) * 4).to_le_bytes());
        wav.extend_from_slice(&(channels * 4).to_le_bytes());
        wav.extend_from_slice(&32u16.to_le_bytes());
        // data chunk
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for &s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        wav
    }
}
