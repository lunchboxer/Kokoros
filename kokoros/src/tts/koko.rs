use crate::onn::ort_koko::{self};
use crate::tts::tokenize::tokenize;
use crate::utils::debug::format_debug_prefix;
use lazy_static::lazy_static;
use ndarray::Array3;
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};

use espeak_rs::text_to_phonemes;

// Global mutex to serialize espeak-rs calls to prevent phoneme randomization
// espeak-rs uses global state internally and is not thread-safe
lazy_static! {
    static ref ESPEAK_MUTEX: Mutex<()> = Mutex::new(());
}

#[derive(Debug, Clone)]
pub struct TTSOpts<'a> {
    pub txt: &'a str,
    pub lan: &'a str,
    pub style_name: &'a str,
    pub save_path: &'a str,
    pub mono: bool,
    pub speed: f32,
    pub initial_silence: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TTSRawAudioOpts<'a> {
    pub txt: &'a str,
    pub lan: &'a str,
    pub style_name: &'a str,
    pub speed: f32,
    pub initial_silence: Option<usize>,
    pub request_id: Option<&'a str>,
    pub instance_id: Option<&'a str>,
    pub chunk_number: Option<usize>,
}

#[derive(Clone)]
pub struct TTSKoko {
    #[allow(dead_code)]
    model_path: String,
    model: Arc<Mutex<ort_koko::OrtKoko>>,
    styles: HashMap<String, Vec<[[f32; 256]; 1]>>,
    init_config: InitConfig,
}

#[derive(Clone)]
pub struct InitConfig {
    pub model_url: String,
    pub voices_url: String,
    pub sample_rate: u32,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            model_url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx".into(),
            voices_url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin".into(),
            sample_rate: 24000,
        }
    }
}

impl TTSKoko {
    pub fn new(model_path: &str, voices_path: &str) -> Self {
        Self::from_config(model_path, voices_path, InitConfig::default())
    }

    /// Find file in standard locations
    fn find_file_in_standard_locations(file_path: &str, file_type: &str) -> String {
        // If the provided path exists, use it as-is
        if Path::new(file_path).exists() {
            return file_path.to_string();
        }

        // Get the file name from the path
        let file_name = Path::new(file_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(file_path);

        // Define standard search paths in order of preference
        let search_paths = match file_type {
            "model" => vec![
                // User-specific data directory
                format!(
                    "{}/.local/share/koko/{}",
                    env::var("HOME").unwrap_or_else(|_| ".".to_string()),
                    file_name
                ),
                // System-wide data directories
                format!("/usr/local/share/koko/{}", file_name),
                format!("/usr/share/koko/{}", file_name),
                // Current behavior as fallback
                file_path.to_string(),
            ],
            "voices" => vec![
                // User-specific data directory
                format!(
                    "{}/.local/share/koko/{}",
                    env::var("HOME").unwrap_or_else(|_| ".".to_string()),
                    file_name
                ),
                // System-wide data directories
                format!("/usr/local/share/koko/{}", file_name),
                format!("/usr/share/koko/{}", file_name),
                // Current behavior as fallback
                file_path.to_string(),
            ],
            _ => vec![file_path.to_string()],
        };

        // Return the first path that exists
        for path in search_paths {
            if Path::new(&path).exists() {
                tracing::info!("Found {} file at: {}", file_type, path);
                return path;
            }
        }

        // If none exist, return the original path for error handling upstream
        tracing::warn!(
            "{} file not found in standard locations, using provided path: {}",
            file_type,
            file_path
        );
        file_path.to_string()
    }

    /// Find voices file in standard locations
    fn find_voices_file(voices_path: &str) -> String {
        Self::find_file_in_standard_locations(voices_path, "voices")
    }

    /// Find model file in standard locations
    fn find_model_file(model_path: &str) -> String {
        Self::find_file_in_standard_locations(model_path, "model")
    }

    pub fn from_config(model_path: &str, voices_path: &str, cfg: InitConfig) -> Self {
        // Find model file in standard locations
        let resolved_model_path = Self::find_model_file(model_path);

        if !Path::new(&resolved_model_path).exists() {
            eprintln!("Model file not found: {}", resolved_model_path);
            eprintln!("Please download the model file from: {}", cfg.model_url);
            eprintln!("And place it at one of these locations:");
            eprintln!("  - {}", resolved_model_path);
            eprintln!("  - ~/.local/share/koko/kokoro-v1.0.onnx");
            eprintln!("  - /usr/local/share/koko/kokoro-v1.0.onnx");
            eprintln!("  - /usr/share/koko/kokoro-v1.0.onnx");
            std::process::exit(1);
        }

        // Find voices file in standard locations
        let resolved_voices_path = Self::find_voices_file(voices_path);

        if !Path::new(&resolved_voices_path).exists() {
            eprintln!("Voices data file not found: {}", resolved_voices_path);
            eprintln!(
                "Please download the voices data file from: {}",
                cfg.voices_url
            );
            eprintln!("And place it at one of these locations:");
            eprintln!("  - {}", resolved_voices_path);
            eprintln!("  - ~/.local/share/koko/voices-v1.0.bin");
            eprintln!("  - /usr/local/share/koko/voices-v1.0.bin");
            eprintln!("  - /usr/share/koko/voices-v1.0.bin");
            std::process::exit(1);
        }

        let model = Arc::new(Mutex::new(
            ort_koko::OrtKoko::new(resolved_model_path.to_string())
                .expect("Failed to create Kokoro TTS model"),
        ));
        // TODO: if(not streaming) { model.print_info(); }
        // model.print_info();

        let styles = Self::load_voices(&resolved_voices_path);

        TTSKoko {
            model_path: model_path.to_string(),
            model,
            styles,
            init_config: cfg,
        }
    }

    fn split_text_into_chunks(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let mut chunks = Vec::new();

        // First split by sentences - using common sentence ending punctuation
        let sentences: Vec<&str> = text
            .split(['.', '?', '!', ';'])
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut current_chunk = String::new();

        for sentence in sentences {
            // Clean up the sentence and add back punctuation
            let sentence = format!("{}.", sentence.trim());

            // Convert to phonemes to check token count
            let sentence_phonemes = {
                let _guard = ESPEAK_MUTEX.lock().unwrap();
                text_to_phonemes(&sentence, "en", None, true, false)
                    .unwrap_or_default()
                    .join("")
            };
            let token_count = tokenize(&sentence_phonemes).len();

            if token_count > max_tokens {
                // If single sentence is too long, split by words
                let words: Vec<&str> = sentence.split_whitespace().collect();
                let mut word_chunk = String::new();

                for word in words {
                    let test_chunk = if word_chunk.is_empty() {
                        word.to_string()
                    } else {
                        format!("{} {}", word_chunk, word)
                    };

                    let test_phonemes = {
                        let _guard = ESPEAK_MUTEX.lock().unwrap();
                        text_to_phonemes(&test_chunk, "en", None, true, false)
                            .unwrap_or_default()
                            .join("")
                    };
                    let test_tokens = tokenize(&test_phonemes).len();

                    if test_tokens > max_tokens {
                        if !word_chunk.is_empty() {
                            chunks.push(word_chunk);
                        }
                        word_chunk = word.to_string();
                    } else {
                        word_chunk = test_chunk;
                    }
                }

                if !word_chunk.is_empty() {
                    chunks.push(word_chunk);
                }
            } else if !current_chunk.is_empty() {
                // Try to append to current chunk
                let test_text = format!("{} {}", current_chunk, sentence);
                let test_phonemes = {
                    let _guard = ESPEAK_MUTEX.lock().unwrap();
                    text_to_phonemes(&test_text, "en", None, true, false)
                        .unwrap_or_default()
                        .join("")
                };
                let test_tokens = tokenize(&test_phonemes).len();

                if test_tokens > max_tokens {
                    // If combining would exceed limit, start new chunk
                    chunks.push(current_chunk);
                    current_chunk = sentence;
                } else {
                    current_chunk = test_text;
                }
            } else {
                current_chunk = sentence;
            }
        }

        // Add the last chunk if not empty
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    pub fn tts_raw_audio_opts(
        &self,
        opts: TTSRawAudioOpts,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.tts_raw_audio(
            opts.txt,
            opts.lan,
            opts.style_name,
            opts.speed,
            opts.initial_silence,
            opts.request_id,
            opts.instance_id,
            opts.chunk_number,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tts_raw_audio(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Split text into appropriate chunks
        let chunks = self.split_text_into_chunks(txt, 500); // Using 500 to leave 12 tokens of margin
        let mut final_audio = Vec::new();

        for chunk in chunks {
            // Convert chunk to phonemes
            let phonemes = {
                let _guard = ESPEAK_MUTEX.lock().unwrap();
                text_to_phonemes(&chunk, lan, None, true, false)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?
                    .join("")
            };
            let debug_prefix = format_debug_prefix(request_id, instance_id);
            let chunk_info = chunk_number
                .map(|n| format!("Chunk: {}, ", n))
                .unwrap_or_default();
            tracing::debug!(
                "{} {}text: '{}' -> phonemes: '{}'",
                debug_prefix,
                chunk_info,
                chunk,
                phonemes
            );
            let mut tokens = tokenize(&phonemes);

            for _ in 0..initial_silence.unwrap_or(0) {
                tokens.insert(0, 30);
            }

            // Get style vectors once
            let styles = self.mix_styles(style_name, tokens.len())?;

            // pad a 0 to start and end of tokens
            let mut padded_tokens = vec![0];
            for &token in &tokens {
                padded_tokens.push(token);
            }
            padded_tokens.push(0);

            let tokens = vec![padded_tokens];

            match self.model.lock().unwrap().infer(
                tokens,
                styles.clone(),
                speed,
                request_id,
                instance_id,
                chunk_number,
            ) {
                Ok(chunk_audio) => {
                    let chunk_audio: Vec<f32> = chunk_audio.iter().cloned().collect();
                    final_audio.extend_from_slice(&chunk_audio);
                }
                Err(e) => {
                    eprintln!("Error processing chunk: {:?}", e);
                    eprintln!("Chunk text was: {:?}", chunk);
                    return Err(Box::new(std::io::Error::other(format!(
                        "Chunk processing failed: {:?}",
                        e
                    ))));
                }
            }
        }

        Ok(final_audio)
    }

    pub fn tts(
        &self,
        TTSOpts {
            txt,
            lan,
            style_name,
            save_path,
            mono,
            speed,
            initial_silence,
        }: TTSOpts,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let audio = self.tts_raw_audio_opts(TTSRawAudioOpts {
            txt,
            lan,
            style_name,
            speed,
            initial_silence,
            request_id: None,
            instance_id: None,
            chunk_number: None,
        })?;

        // Save to file
        if mono {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: self.init_config.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = hound::WavWriter::create(save_path, spec)?;
            for &sample in &audio {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        } else {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: self.init_config.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = hound::WavWriter::create(save_path, spec)?;
            for &sample in &audio {
                writer.write_sample(sample)?;
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        eprintln!("Audio saved to {}", save_path);
        Ok(())
    }

    pub fn mix_styles(
        &self,
        style_name: &str,
        tokens_len: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if !style_name.contains("+") {
            if let Some(style) = self.styles.get(style_name) {
                let styles = vec![style[tokens_len][0].to_vec()];
                Ok(styles)
            } else {
                Err(format!("can not found from styles_map: {}", style_name).into())
            }
        } else {
            eprintln!("parsing style mix");
            let styles: Vec<&str> = style_name.split('+').collect();

            let mut style_names = Vec::new();
            let mut style_portions = Vec::new();

            for style in styles {
                if let Some((name, portion)) = style.split_once('.')
                    && let Ok(portion) = portion.parse::<f32>()
                {
                    style_names.push(name);
                    style_portions.push(portion * 0.1);
                }
            }
            eprintln!("styles: {:?}, portions: {:?}", style_names, style_portions);

            let mut blended_style = vec![vec![0.0; 256]; 1];

            for (name, portion) in style_names.iter().zip(style_portions.iter()) {
                if let Some(style) = self.styles.get(*name) {
                    let style_slice = &style[tokens_len][0]; // This is a [256] array
                    // Blend into the blended_style
                    for (j, &value) in style_slice.iter().enumerate().take(256) {
                        blended_style[0][j] += value * portion;
                    }
                }
            }
            Ok(blended_style)
        }
    }

    fn load_voices(voices_path: &str) -> HashMap<String, Vec<[[f32; 256]; 1]>> {
        let mut npz = NpzReader::new(File::open(voices_path).unwrap()).unwrap();
        let mut map = HashMap::new();

        for voice in npz.names().unwrap() {
            let voice_data: Result<Array3<f32>, _> = npz.by_name(&voice);
            let voice_data = voice_data.unwrap();
            let mut tensor = vec![[[0.0; 256]; 1]; 511];
            for (i, inner_value) in voice_data.outer_iter().enumerate() {
                for (j, inner_inner_value) in inner_value.outer_iter().enumerate() {
                    for (k, number) in inner_inner_value.iter().enumerate() {
                        tensor[i][j][k] = *number;
                    }
                }
            }
            map.insert(voice, tensor);
        }

        // Sort voices for consistent ordering
        let _sorted_voices = {
            let mut voices = map.keys().collect::<Vec<_>>();
            voices.sort();
            voices
        };

        map
    }

    // Returns a sorted list of available voice names
    pub fn get_available_voices(&self) -> Vec<String> {
        let mut voices: Vec<String> = self.styles.keys().cloned().collect();
        voices.sort();
        voices
    }
}
