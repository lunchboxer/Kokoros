<div align="center">
  <img src="https://img2023.cnblogs.com/blog/3572323/202501/3572323-20250112184100378-907988670.jpg" alt="Banner" width="400" height="190">
</div>
<br>
<h1 align="center">Kokoro Rust</h1>

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) is the best available open-source TTS model. This repo provides **insanely fast Kokoro infer in Rust**. Build and enjoy the `koko` command-line utility.

`kokoros` is a `rust` crate that provides easy to use TTS ability.
One can directly call `koko` in terminal to synthesize audio.

`kokoros` uses a relative small model 87M params while providing high quality voices results.

## Installation

1. Build the project and download required data files:

```bash
make all
```

This will:
- Build the project in release mode
- Download the Kokoro ONNX model (`checkpoints/kokoro-v1.0.onnx`)
- Download the voices data file (`data/voices-v1.0.bin`)
- Verify the integrity of downloaded files

3. (Optional) Install the binary and voice data:

For user-wide installation (default):
```bash
make install
```

For system-wide installation:
```bash
sudo make install INSTALL_TYPE=system
```

This will install:
- The `koko` binary to `~/.local/bin` (user) or `/usr/local/bin` (system)
- The voice data to `~/.local/share/koko` (user) or `/usr/local/share/koko` (system)

The `koko` binary can be used without installation at `./target/release/koko`.

## Usage

### View available options

```bash
koko -h
```

### Generate speech for some text

```
koko text "Hello, this is a TTS test"
```

The generated audio will be saved to `./output.wav` by default. You can customize the save location with the `--output` or `-o` option:

```
koko text "I hope you're having a great day today!" --output greeting.wav
```

### Generate speech for each line in a file

```
koko file poem.txt
```

For a file with 3 lines of text, by default, speech audio files `./output_0.wav`, `./output_1.wav`, `./output_2.wav` will be outputted. For files with 10 or more lines, zero-padding is automatically applied to ensure proper alphanumeric sorting (e.g., `./output_00.wav`, `./output_01.wav`, ..., `./output_10.wav`). You can customize the save location with the `--output` or `-o` option, using `{line}` as the line number:

```
koko file lyrics.txt -o "song/lyric_{line}.wav"
```

### Parallel Processing Configuration

Configure parallel TTS instances for the OpenAI-compatible server based on your performance preference:

```
# Best 0.5-2 seconds time-to-first-audio (lowest latency)
koko openai --instances 1

# Balanced performance (default, 2 instances, usually best throughput for CPU processing)
koko openai

# Best total processing time (Diminishing returns on CPU processing observed on Mac M2)
koko openai --instances 4
```

#### How to determine the optimal number of instances for your system configuration?

Choose your configuration based on use case:
- Single instance for real-time applications requiring immediate audio response irrespective of system configuration.
- Multiple instances for batch processing where total completion time matters more than initial latency.
  - This was benchmarked on a Mac M2 with 8 cores and 24GB RAM.
  - Tested with the message:
    > Welcome to our comprehensive technology demonstration session. Today we will explore advanced parallel processing systems thoroughly. These systems utilize multiple computational instances simultaneously for efficiency. Each instance processes different segments concurrently without interference. The coordination between instances ensures seamless output delivery consistently. Modern algorithms optimize resource utilization effectively across all components. Performance improvements are measurable and significant in real scenarios. Quality assurance validates each processing stage thoroughly before deployment. Integration testing confirms system reliability consistently under various conditions. User experience remains smooth throughout operation regardless of complexity. Advanced monitoring tracks system performance metrics continuously during execution.
  - Benchmark results (avg of 5)
    | No. of instances | TTFA | Total time |
    |------------------|------|------------|
    | 1                | 1.44s | 19.0s     |
    | 2                | 2.44s | 16.1s     |
    | 4                | 4.98s | 16.6s     |
  - If you have a CPU, memory bandwidth will be the usual bottleneck. You will have to experiment to find a sweet spot of number of instances giving you optimal throughput on your system configuration.
  - If you have a NVIDIA GPU, you can try increasing the number of instances. You are expected to further improve throughput.
  - Attempts to [make this work on CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html), would likely start with converting the ONNX model to CoreML or ORT.

*Note: The `--instances` flag is currently supported in API server mode. CLI text commands will support parallel processing in future releases.*

### Streaming

The `stream` option will start the program, reading for lines of input from stdin and outputting WAV audio to stdout.

Use it in conjunction with piping.

#### Typing manually

```
koko stream > live-audio.wav
# Start typing some text to generate speech for and hit enter to submit
# Speech will append to `live-audio.wav` as it is generated
# Hit Ctrl D to exit
```

#### Input from another source

```
echo "Suppose some other program was outputting lines of text" | koko stream > programmatic-audio.wav
```

### With docker

1. Build the image

```bash
docker build -t kokoros .
```

2. Run the image, passing options as described above

```bash
# Basic text to speech
docker run -v ./tmp:/app/tmp kokoros text "Hello from docker!" -o tmp/hello.wav
```

## Copyright & License

This project is a fork of the original **Kokoros** [Kokoros](https://github.com/lucas-jin/kokoros) repository by Lucas Jin. All core functionality is based on his foundational work.

This software is released under the **Apache License, Version 2.0**. A full copy of the license is included in this repository.
