use std::borrow::Cow;

use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ort::execution_providers::cpu::CPUExecutionProvider;
#[cfg(feature = "cuda")]
use ort::execution_providers::cuda::CUDAExecutionProvider;
use ort::logging::LogLevel;
use ort::session::builder::SessionBuilder;
use ort::{
    session::{Session, SessionInputValue, SessionInputs, SessionOutputs},
    value::{Tensor, Value},
};

use crate::utils::debug::format_debug_prefix;

pub struct KokoroModel {
    sess: Session,
}

impl KokoroModel {
    pub fn new(model_path: String) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        let providers = [CUDAExecutionProvider::default().build()];

        #[cfg(not(feature = "cuda"))]
        let providers = [CPUExecutionProvider::default().build()];

        let session = SessionBuilder::new()
            .map_err(|e| format!("Failed to create session builder: {}", e))?
            .with_execution_providers(providers)
            .map_err(|e| format!("Failed to build session: {}", e))?
            .with_log_level(LogLevel::Warning)
            .map_err(|e| format!("Failed to set log level: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to commit from file: {}", e))?;

        Ok(KokoroModel { sess: session })
    }

    pub fn print_info(&self) {
        eprintln!("Input names:");
        for input in &self.sess.inputs {
            eprintln!("  - {}", input.name);
        }
        eprintln!("Output names:");
        for output in &self.sess.outputs {
            eprintln!("  - {}", output.name);
        }

        #[cfg(feature = "cuda")]
        eprintln!("Configured with: CUDA execution provider");

        #[cfg(not(feature = "cuda"))]
        eprintln!("Configured with: CPU execution provider");
    }

    pub fn infer(
        &mut self,
        tokens: Vec<Vec<i64>>,
        styles: Vec<Vec<f32>>,
        speed: f32,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, Box<dyn std::error::Error>> {
        let shape = [tokens.len(), tokens[0].len()];
        let tokens_flat: Vec<i64> = tokens.into_iter().flatten().collect();

        let debug_prefix = format_debug_prefix(request_id, instance_id);
        let chunk_info = chunk_number
            .map(|n| format!("Chunk: {}, ", n))
            .unwrap_or_default();
        tracing::debug!(
            "{} {}inference input: tokens_shape={:?}, tokens_count={}, styles_shape={:?}",
            debug_prefix,
            chunk_info,
            shape,
            tokens_flat.len(),
            [styles.len(), styles[0].len()]
        );

        let tokens = Tensor::from_array((shape, tokens_flat))?;
        let tokens_value: SessionInputValue = SessionInputValue::Owned(Value::from(tokens));

        let shape_style = [styles.len(), styles[0].len()];
        let style_flat: Vec<f32> = styles.into_iter().flatten().collect();
        let style = Tensor::from_array((shape_style, style_flat))?;
        let style_value: SessionInputValue = SessionInputValue::Owned(Value::from(style));

        let speed = vec![speed; 1];
        let speed = Tensor::from_array(([1], speed))?;
        let speed_value: SessionInputValue = SessionInputValue::Owned(Value::from(speed));

        let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (Cow::Borrowed("tokens"), tokens_value),
            (Cow::Borrowed("style"), style_value),
            (Cow::Borrowed("speed"), speed_value),
        ];

        let outputs: SessionOutputs = self.sess.run(SessionInputs::from(inputs))?;
        let (shape, data) = outputs["audio"]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        // Convert Shape and &[f32] to ArrayBase<OwnedRepr<f32>, IxDyn>
        let shape_vec: Vec<usize> = shape.iter().map(|&i| i as usize).collect();
        let data_vec: Vec<f32> = data.to_vec();
        let debug_prefix = format_debug_prefix(request_id, instance_id);
        let chunk_info = chunk_number
            .map(|n| format!("Chunk: {}, ", n))
            .unwrap_or_default();
        tracing::debug!(
            "{} {}inference output: audio_shape={:?}, sample_count={}",
            debug_prefix,
            chunk_info,
            shape_vec,
            data_vec.len()
        );
        let output_array =
            ArrayBase::<OwnedRepr<f32>, IxDyn>::from_shape_vec(shape_vec, data_vec)?;

        Ok(output_array)
    }
}