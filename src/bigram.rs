use rand::{distributions::WeightedIndex, prelude::*, Rng};

use crate::codec::{Codec, Decoder, Encoder, CONTROL, DOT_CONTROL};
use std::collections::BTreeMap;

const NAMES_DATASET: &str = include_str!("../makemore/names.txt");
const N_DIMS: usize = 28; // Length of our vocabulary, referenced in multiple places

type BigramPair = (char, char);

struct Bigram<'a> {
    dataset: &'a Vec<Vec<char>>,
    pub(crate) N: [[u32; N_DIMS]; N_DIMS],
    pub(crate) P: [[f32; N_DIMS]; N_DIMS], // Matrix of normalised probabilities
    codec: Codec,
}

impl<'a> Bigram<'a> {
    fn new(dataset: &'a Vec<Vec<char>>) -> Self {
        Self {
            dataset,
            N: [[0; N_DIMS]; N_DIMS],
            P: [[0.0; N_DIMS]; N_DIMS],
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

                self.N[i_ch1][i_ch2] += 1;
            }
        }

        // We need to compute row-rise sums and produce a single (N_DIMS , 1) tensor
        //
        // Now since our `P` matrix is (N_DIMS, N_DIMS) shape, we need to align the
        // `row_sums` to match that in order to normalise our values. In pytorch, this
        // is done automatically as part of broadcasting rules. Here we implement it
        // directly.
        let row_sums: Vec<f32> = self
            .N
            .iter()
            .map(|r| f32::from_bits(r.into_iter().sum()))
            .collect();

        // We want to compute the probabilities once, and reference into them
        // rather than normalising them each time.
        //
        // NOTE(juxhin): These normalised rows are f32s, and therefore lose some
        // precision. Many times resulting in 0.999993 or 1.000001. Also since we
        // have our control char on 28th dimension that is never used, that results
        // in a NaN.
        for i in 0..N_DIMS {
            let normalised_row = self.N[i]
                .iter()
                .map(|cell| f32::from_bits(*cell) / row_sums.get(i).unwrap())
                .collect::<Vec<f32>>();
            self.P[i] = Self::vec_to_bounded_slice(normalised_row);
        }
    }

    fn vec_to_bounded_slice<T, const N: usize>(v: Vec<T>) -> [T; N] {
        v.try_into().unwrap_or_else(|v: Vec<T>| {
            panic!("Expected a Vec of length {} but it was {}", N, v.len())
        })
    }

    fn sample(&self) -> Vec<char> {
        let mut out = vec![];
        let mut rng = rand::thread_rng();

        let mut sample = 0;

        loop {
            let row = self.P[sample];

            let dist = WeightedIndex::new(&row).unwrap();
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
