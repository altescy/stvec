use super::tokenizer::tokenize;
use super::vocab::{Vocabulary, VocabularyParams};
use anyhow::Result;
use hashbrown::HashSet;
use numpy::PyArray2;
use pyo3::prelude::*;

#[pyclass(name = "Indexer")]
pub struct Indexer {
    vocabulary: Vocabulary,
}

type IndexerParams = VocabularyParams;

#[pymethods]
impl Indexer {
    #[new]
    fn __new__(min_df: usize, max_df: usize, stop_words: HashSet<String>) -> Self {
        Indexer {
            vocabulary: Vocabulary::new(min_df, max_df, true, stop_words),
        }
    }

    fn bos(&self) -> usize {
        self.vocabulary.bos().unwrap()
    }

    fn eos(&self) -> usize {
        self.vocabulary.eos().unwrap()
    }

    fn pad(&self) -> usize {
        self.vocabulary.pad().unwrap()
    }

    fn unk(&self) -> usize {
        self.vocabulary.unk().unwrap()
    }

    fn train(&mut self, docs: Vec<String>) {
        self.vocabulary.train(&docs);
    }

    fn vectorize<'py>(
        &self,
        py: Python<'py>,
        docs: Vec<String>,
    ) -> Result<(&'py PyArray2<usize>, &'py PyArray2<bool>)> {
        let mut max_length: usize = 0;
        let mut indices: Vec<Vec<usize>> = vec![];
        let mut mask: Vec<Vec<bool>> = vec![];
        let bos_index = self.bos();
        let eos_index = self.eos();
        let unk_index = self.unk();
        let pad_index = self.pad();
        for d in 0..docs.len() {
            mask.push(vec![]);
            indices.push(vec![]);
            mask[d].push(true);
            indices[d].push(bos_index);
            for token in tokenize(&docs[d]) {
                mask[d].push(true);
                if let Some((index, _)) = self.vocabulary.get(token) {
                    indices[d].push(*index);
                } else {
                    indices[d].push(unk_index);
                }
            }
            mask[d].push(true);
            indices[d].push(eos_index);
            let length = indices[0].len();
            if length > max_length {
                max_length = length;
            }
        }
        for d in 0..docs.len() {
            let diff = max_length - indices[d].len();
            for _ in 0..diff {
                indices[d].push(pad_index);
                mask[d].push(false);
            }
        }
        let indices = PyArray2::from_vec2(py, &indices)?;
        let mask = PyArray2::from_vec2(py, &mask)?;
        Ok((indices, mask))
    }

    fn to_params(&self) -> IndexerParams {
        self.vocabulary.to_params()
    }

    #[staticmethod]
    fn from_params(params: IndexerParams) -> Indexer {
        let vocab_params = params;
        Indexer {
            vocabulary: Vocabulary::from_params(vocab_params),
        }
    }
}
