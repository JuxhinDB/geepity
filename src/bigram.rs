use rand::{distributions::WeightedIndex, prelude::*, Rng};

use crate::{
    codec::{Codec, Decoder, Encoder, CONTROL, DOT_CONTROL},
    util::{mat_mul, one_hot, square_uniform_matrix, transpose, vec_to_bounded_slice},
};
const NAMES_DATASET: &str = include_str!("../makemore/names.txt");
const N_DIMS: usize = 28; // Length of our vocabulary, referenced in multiple places

type BigramPair = (char, char);

type Dataset = (Vec<Vec<f32>>, Vec<Vec<f32>>); // Maps to inputs and targets/labels

struct Bigram<'a> {
    dataset: &'a Vec<Vec<char>>,
    pub(crate) N: [[u32; N_DIMS]; N_DIMS],
    P: [[f32; N_DIMS]; N_DIMS], // Matrix of normalised probabilities
    W: [[f32; N_DIMS]; N_DIMS], // Our single layer of neurons
    nll: f32,                   // The normalised, negative log likelihood of the model
    training: Dataset,
    testing: Dataset,
    codec: Codec,
}

impl<'a> Bigram<'a> {
    fn new(dataset: &'a Vec<Vec<char>>) -> Self {
        Self {
            dataset,
            N: [[1; N_DIMS]; N_DIMS],
            P: [[0.0; N_DIMS]; N_DIMS],
            W: square_uniform_matrix(),
            nll: 0.0,
            training: (Vec::new(), Vec::new()),
            testing: (Vec::new(), Vec::new()),
            codec: Codec::new(dataset),
        }
    }

    /// Computes the bigram map
    fn compute(&mut self) {
        for name in self.dataset.iter().take(1) {
            let control = [DOT_CONTROL];

            let chars = control.iter().chain(name.iter()).chain(control.iter());

            // FIXME(jdb): remove clone
            for (ch1, ch2) in chars.clone().zip(chars.skip(1)) {
                // Compute `N` matrix
                let i_ch1 = self.codec.encode(ch1);
                let i_ch2 = self.codec.encode(ch2);

                self.training.0.push(one_hot::<N_DIMS>(i_ch1));
                self.training.1.push(one_hot::<N_DIMS>(i_ch2));
            }
        }

        let a = vec_to_bounded_slice::<[f32; N_DIMS], 5>(
            self.training
                .0
                .iter()
                .cloned()
                .map(vec_to_bounded_slice::<f32, N_DIMS>)
                .collect(),
        );

        let mul = mat_mul(a, transpose(self.W));
        for row in mul {
            println!("{row:?}\n");
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
            self.P[i] = vec_to_bounded_slice(normalised_row);
        }

        // Calculate the model's performance, i.e., calculating the normalised
        // negative log probability of the model.
        let mut log_likelihood = 0.0;
        let mut n = 0;
        for name in self.dataset.iter() {
            let control = [DOT_CONTROL];
            let chars = control.iter().chain(name.iter()).chain(control.iter());

            // FIXME(jdb): remove clone
            for (ch1, ch2) in chars.clone().zip(chars.skip(1)) {
                let i_ch1 = self.codec.encode(ch1);
                let i_ch2 = self.codec.encode(ch2);

                let prob = self.P[i_ch1][i_ch2];

                // Rather than computing the product of probabilities, and then
                // calculating the log, we can leverage log properties and
                // simply sum all ln(probabilities).
                log_likelihood += prob.ln();
                n += 1; // Used to normalise the final number
            }
        }

        log_likelihood = log_likelihood.abs() / n as f32;
        self.nll = log_likelihood;
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
