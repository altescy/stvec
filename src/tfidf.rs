use super::vocab::Vocabulary;
use hashbrown::HashMap;
use numpy::PyArray1;
use pyo3::prelude::*;

#[pyclass(name = "TfidfVectorizer")]
pub struct TfidfVectorizer {
    use_idf: bool,
    vocabulary: Vocabulary,
}

fn vectorize(
    texts: &Vec<String>,
    vocab: &Vocabulary,
    use_idf: bool,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row: Vec<usize> = vec![];
    let mut col: Vec<usize> = vec![];
    let mut dat: Vec<f64> = vec![];
    let log_total_docs = (vocab.total_docs() as f64).ln_1p();
    for i in 0..texts.len() {
        let mut count: HashMap<usize, f64> = HashMap::new();
        for token in texts[i].trim().split_whitespace() {
            if let Some((index, df)) = vocab.get(token) {
                let weight = if use_idf {
                    log_total_docs - (*df as f64).ln_1p() + 1.0
                } else {
                    *df as f64
                };
                count
                    .entry(*index)
                    .and_modify(|value| *value += weight)
                    .or_insert(1.0);
            }
        }
        for (index, value) in count.into_iter() {
            row.push(i);
            col.push(index);
            dat.push(value);
        }
    }
    (row, col, dat)
}

#[pymethods]
impl TfidfVectorizer {
    #[new]
    fn __new__(min_df: usize, max_df: usize, use_idf: bool) -> Self {
        TfidfVectorizer {
            use_idf: use_idf,
            vocabulary: Vocabulary::new(min_df, max_df),
        }
    }

    fn get_output_dim(&self) -> usize {
        self.vocabulary.len()
    }

    fn train(&mut self, docs: Vec<String>) {
        self.vocabulary.train(&docs);
    }

    fn vectorize<'py>(
        &self,
        py: Python<'py>,
        docs: Vec<String>,
    ) -> (
        &'py PyArray1<usize>,
        &'py PyArray1<usize>,
        &'py PyArray1<f64>,
    ) {
        let (row, col, dat) = vectorize(&docs, &self.vocabulary, self.use_idf);
        let row = PyArray1::from_vec(py, row);
        let col = PyArray1::from_vec(py, col);
        let dat = PyArray1::from_vec(py, dat);
        (row, col, dat)
    }

    fn to_params(&self) -> (bool, (usize, usize, usize, Vec<(String, usize, usize)>)) {
        (self.use_idf, self.vocabulary.to_params())
    }

    #[staticmethod]
    fn from_params(params: (bool, (usize, usize, usize, Vec<(String, usize, usize)>))) -> Self {
        let (use_idf, vocab_params) = params;
        TfidfVectorizer {
            use_idf: use_idf,
            vocabulary: Vocabulary::from_params(vocab_params),
        }
    }
}
