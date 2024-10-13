use rand::{distributions::WeightedIndex, prelude::*, Rng};

use crate::codec::{Codec, Decoder, Encoder, CONTROL, DOT_CONTROL};
use std::collections::BTreeMap;

const NAMES_DATASET: &str = include_str!("../makemore/names.txt");

type BigramPair = (char, char);

struct Bigram<'a> {
    dataset: &'a Vec<Vec<char>>,
    pub(crate) N: [[u32; 28]; 28],
    codec: Codec,
}

impl<'a> Bigram<'a> {
    fn new(dataset: &'a Vec<Vec<char>>) -> Self {
        Self {
            dataset,
            N: [[0; 28]; 28],
            codec: Codec::new(dataset),
        }
    }

    /// Computes the bigram map
    fn compute(&mut self) {
        for name in self.dataset.iter() {
            let control = [DOT_CONTROL];

            let chars = control.iter().chain(name.iter()).chain(control.iter());

            // FIXME(jdb): remove clone
            for (ch1, ch2) in chars.clone().zip(chars.skip(1)) {
                // Compute `N` matrix
                let i_ch1 = self.codec.encode(ch1);
                let i_ch2 = self.codec.encode(ch2);

                self.N[i_ch1][i_ch2] += 1
            }
        }
    }

    fn sample(&self) -> Vec<char> {
        let mut out = vec![];
        let mut rng = rand::thread_rng();

        let mut sample = 0;

        loop {
            let probabilities: Vec<f32> = self.N[sample][..]
                .iter()
                .map(|count| f32::from_bits(*count))
                .collect();

            let probabilities_sum: f32 = probabilities.iter().sum();

            let normalised: Vec<f32> = probabilities
                .iter()
                .map(|p| p / probabilities_sum)
                .collect();

            let dist = WeightedIndex::new(&normalised).unwrap();
            sample = dist.sample(&mut rng);

            if sample == 0 {
                break;
            }

            out.push(self.codec.decode(sample as u16));
        }

        out
    }
}

mod tests {
    use super::*;

    #[test]
    fn test() {
        let dataset: Vec<Vec<char>> = NAMES_DATASET
            .split("\n")
            .map(|name| name.chars().collect())
            .collect();

        let mut bigram = Bigram::new(&dataset);
        bigram.compute();

        println!("name: {:?}", bigram.sample());
    }
}
