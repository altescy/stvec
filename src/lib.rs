use pyo3::prelude::*;

mod indexer;
mod tfidf;
mod tokenizer;
mod vocab;

#[pymodule]
fn stvec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tfidf::TfidfVectorizer>()?;
    m.add_class::<indexer::Indexer>()?;
    Ok(())
}
