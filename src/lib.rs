use pyo3::prelude::*;

mod tfidf;
mod vocab;

#[pymodule]
fn stvec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tfidf::TfidfVectorizer>()?;
    Ok(())
}
