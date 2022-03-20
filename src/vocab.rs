use hashbrown::HashMap;

pub struct Vocabulary {
    min_df: usize,
    max_df: usize,
    total_docs: usize,
    tokens: HashMap<String, (usize, usize)>,
}

impl Vocabulary {
    pub fn new(min_df: usize, max_df: usize) -> Self {
        Vocabulary {
            min_df: min_df,
            max_df: max_df,
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
            for token in docs[d].trim().split_whitespace() {
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
        let mut index: usize = 0;
        for (&token, &(_, df)) in dfs.iter() {
            if self.min_df <= df && df <= self.max_df {
                self.tokens.insert(String::from(token), (index, df));
                index += 1;
            }
        }
    }

    pub fn get(&self, token: &str) -> Option<&(usize, usize)> {
        self.tokens.get(token)
    }

    pub fn total_docs(&self) -> usize {
        self.total_docs
    }

    pub fn to_params(&self) -> (usize, usize, usize, Vec<(String, usize, usize)>) {
        (
            self.min_df,
            self.max_df,
            self.total_docs,
            self.tokens
                .iter()
                .map(|(token, &(index, df))| (String::from(token), index, df))
                .collect(),
        )
    }

    pub fn from_params(params: (usize, usize, usize, Vec<(String, usize, usize)>)) -> Self {
        let (min_df, max_df, total_docs, tokens) = params;
        Vocabulary {
            min_df: min_df,
            max_df: max_df,
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

        let mut vocab = Vocabulary::new(0, 10);
        vocab.train(&texts);
        assert_eq!(vocab.get("this").unwrap().1, 2);
        assert_eq!(vocab.get("first").unwrap().1, 1);
    }
}
