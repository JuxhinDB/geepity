use std::collections::BTreeMap;

const NAMES_DATASET: &str = include_str!("../makemore/names.txt");

type BigramPair = (char, char);

struct Bigram<'a> {
    dataset: &'a Vec<Vec<char>>,
    pub(crate) bigram: BTreeMap<BigramPair, usize>,
    N: [[u32; 28]; 28],
}

const SEQ_START: char = '\u{0000}';
const SEQ_END: char = '\u{ffff}';

impl<'a> Bigram<'a> {
    fn new(dataset: &'a Vec<Vec<char>>) -> Self {
        // Represents our count matrix. It's 26 characters of the alphabet
        // plus the two control characters \u0000 and \uffff.
        let N = [[0; 28]; 28];

        Self {
            dataset,
            bigram: BTreeMap::new(),
            N,
        }
    }

    /// Computes the bigram map
    fn compute(&mut self) {
        let seq_start = [SEQ_START];
        let seq_end = [SEQ_END];

        for word in self.dataset.iter() {
            let chars = seq_start.iter().chain(word.iter()).chain(seq_end.iter());

            // FIXME(jdb): remove clone
            for (ch1, ch2) in chars.clone().zip(chars.skip(1)) {
                let bigram = (*ch1, *ch2);

                let count = self.bigram.get(&bigram).unwrap_or(&0) + 1;
                self.bigram.insert(bigram, count);
            }
        }
    }

    fn 
}

mod tests {
    use super::*;

    #[test]
    fn test_sliding() {
        let dataset: Vec<Vec<char>> = NAMES_DATASET
            .split("\n")
            .map(|w| w.chars().collect())
            .collect();

        let mut bigram = Bigram::new(&dataset);
        bigram.compute();

        for (k, v) in bigram.bigram.iter() {
            println!("{k:?}: {v:?}");
        }
    }
}
