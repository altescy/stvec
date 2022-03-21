use super::vocab::Vocabulary;
use anyhow::Result;
use numpy::PyArray2;
use pyo3::prelude::*;

#[pyclass(name = "Indexer")]
pub struct Indexer {
    vocabulary: Vocabulary,
}

#[pymethods]
impl Indexer {
    #[new]
    fn __new__(min_df: usize, max_df: usize) -> Self {
        Indexer {
            vocabulary: Vocabulary::new(min_df, max_df, true),
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
        let unk_index = self.unk();
        for d in 0..docs.len() {
            mask.push(vec![]);
            indices.push(vec![]);
            let mut length = 0;
            for token in docs[d].trim().split_whitespace() {
                length += 1;
                mask[d].push(true);
                if let Some((index, _)) = self.vocabulary.get(token) {
                    indices[d].push(*index);
                } else {
                    indices[d].push(unk_index);
                }
            }
            if length > max_length {
                max_length = length;
            }
        }
        let pad_index = self.pad();
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
}
