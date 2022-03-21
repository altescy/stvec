use super::tokenizer::tokenize;
use hashbrown::{HashMap, HashSet};

const BOS: &'static str = "@@BOS@@";
const EOS: &'static str = "@@EOS@@";
const PAD: &'static str = "@@PAD@@";
const UNK: &'static str = "@@UNK@@";

pub struct Vocabulary {
    min_df: usize,
    max_df: usize,
    use_specials: bool,
    stop_words: HashSet<String>,
    total_docs: usize,
    tokens: HashMap<String, (usize, usize)>,
}

pub type VocabularyParams = (
    usize,
    usize,
    bool,
    HashSet<String>,
    usize,
    Vec<(String, usize, usize)>,
);

impl Vocabulary {
    pub fn new(
        min_df: usize,
        max_df: usize,
        use_specials: bool,
        stop_words: HashSet<String>,
    ) -> Self {
        Vocabulary {
            min_df: min_df,
            max_df: max_df,
            use_specials: use_specials,
            stop_words: stop_words,
            total_docs: 0,
            tokens: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn train(&mut self, docs: &Vec<String>) {
        self.tokens.clear();
        self.total_docs = docs.len();
        let mut dfs = HashMap::new();
        for d in 0..docs.len() {
            for token in tokenize(&docs[d]) {
                dfs.entry(token)
                    .and_modify(|(prev, df)| {
                        if *prev != d {
                            *prev = d;
                            *df += 1usize;
                        }
                    })
                    .or_insert((d, 1usize));
            }
        }
        if self.use_specials {
            self.tokens
                .insert(String::from(BOS), (self.tokens.len(), 0));
            self.tokens
                .insert(String::from(EOS), (self.tokens.len(), 0));
            self.tokens
                .insert(String::from(PAD), (self.tokens.len(), 0));
            self.tokens
                .insert(String::from(UNK), (self.tokens.len(), 0));
        }
        let mut index: usize = self.tokens.len();
        for (&token, &(_, df)) in dfs.iter() {
            if self.stop_words.contains(token) {
                continue;
            }
            if self.min_df <= df && df <= self.max_df {
                self.tokens.insert(String::from(token), (index, df));
                index += 1;
            }
        }
    }

    pub fn get(&self, token: &str) -> Option<&(usize, usize)> {
        if self.stop_words.contains(token) {
            None
        } else {
            self.tokens.get(token)
        }
    }

    pub fn bos(&self) -> Option<usize> {
        self.tokens.get(BOS).map(|(index, _)| *index)
    }

    pub fn eos(&self) -> Option<usize> {
        self.tokens.get(EOS).map(|(index, _)| *index)
    }

    pub fn pad(&self) -> Option<usize> {
        self.tokens.get(PAD).map(|(index, _)| *index)
    }

    pub fn unk(&self) -> Option<usize> {
        self.tokens.get(UNK).map(|(index, _)| *index)
    }

    pub fn total_docs(&self) -> usize {
        self.total_docs
    }

    pub fn to_params(&self) -> VocabularyParams {
        (
            self.min_df,
            self.max_df,
            self.use_specials,
            self.stop_words.clone(),
            self.total_docs,
            self.tokens
                .iter()
                .map(|(token, &(index, df))| (String::from(token), index, df))
                .collect(),
        )
    }

    pub fn from_params(params: VocabularyParams) -> Self {
        let (min_df, max_df, use_specials, stop_words, total_docs, tokens) = params;
        Vocabulary {
            min_df: min_df,
            max_df: max_df,
            use_specials: use_specials,
            stop_words: stop_words,
            total_docs: total_docs,
            tokens: tokens
                .iter()
                .map(|(token, index, df)| (String::from(token), (*index, *df)))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary() {
        let texts = vec![
            String::from("this is a first sentence"),
            String::from("this is a second sentence"),
        ];
        let stop_words = HashSet::from([String::from("is")]);

        let mut vocab = Vocabulary::new(0, 10, false, stop_words);
        vocab.train(&texts);
        assert_eq!(vocab.get("this").unwrap().1, 2);
        assert_eq!(vocab.get("first").unwrap().1, 1);
        assert_eq!(vocab.get("is"), None);
    }
}
