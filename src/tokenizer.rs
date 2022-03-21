use std::str::SplitWhitespace;

pub fn tokenize(text: &str) -> SplitWhitespace {
    text.trim().split_whitespace()
}
