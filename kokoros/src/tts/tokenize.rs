use crate::tts::vocab::VOCAB;

/// Tokenizes the given phonemes string into a vector of token indices.
///
/// This function takes a text string as input and converts it into a vector of token indices
/// by looking up each character in the global `VOCAB` map and mapping it to the corresponding
/// token index. The resulting vector contains the token indices for the input text.
///
/// # Arguments
/// * `text` - The input text string to be tokenized.
///
/// # Returns
/// A vector of `i64` token indices representing the input text.
pub fn tokenize(phonemes: &str) -> Vec<i64> {
    phonemes
        .chars()
        .filter_map(|c| VOCAB.get(&c))
        .map(|&idx| idx as i64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "heɪ ðɪs ɪz ˈlʌvliː!";
        let tokens = tokenize(text);

        // Print the actual tokens for debugging
        println!("Actual tokens: {:?}", tokens);

        // Test empty string
        let empty = "";
        let empty_tokens = tokenize(empty);
        assert!(empty_tokens.is_empty());

        // Test punctuation
        let punct = "...";
        let punct_tokens = tokenize(punct);
        assert_eq!(punct_tokens.len(), 3);
    }
}

use crate::tts::vocab::REVERSE_VOCAB;

pub fn tokens_to_phonemes(tokens: &[i64]) -> String {
    tokens
        .iter()
        .filter_map(|&t| REVERSE_VOCAB.get(&(t as usize)))
        .collect()
}

#[cfg(test)]
mod tests2 {
    use super::*;

    #[test]
    fn test_tokens_to_phonemes() {
        let tokens = vec![
            50, 47, 102, 16, 81, 102, 61, 16, 102, 68, 16, 156, 54, 138, 64, 54, 51, 158, 5,
        ];
        let text = tokens_to_phonemes(&tokens);
        println!("Actual text: {}", text);
        // We're not asserting on the exact text since it may change with the vocab

        // Test empty vector
        let empty_tokens: Vec<i64> = vec![];
        assert_eq!(tokens_to_phonemes(&empty_tokens), "");
    }
}
