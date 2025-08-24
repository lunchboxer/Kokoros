<div align="center">
  <img src="./koko-redux.webp" alt="Banner" width="400">
</div>
<br>
<h1 align="center">Koko-redux</h1>

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
