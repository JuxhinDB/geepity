use rand::{distributions::WeightedIndex, prelude::*, Rng};

use crate::{
    codec::{Codec, Decoder, Encoder, CONTROL, DOT_CONTROL},
    util::{exponentiate, mat_mul, normalize, one_hot, square_uniform_probability_matrix, transpose, vec_to_bounded_slice},
};
const NAMES_DATASET: &str = include_str!("../makemore/names.txt");
const N_DIMS: usize = 28; // Length of our vocabulary, referenced in multiple places

type BigramPair = (char, char);

type Dataset = (Vec<Vec<f32>>, Vec<Vec<f32>>); // Maps to inputs and targets/labels

struct Bigram<'a> {
    dataset: &'a Vec<Vec<char>>,
    P: [[f32; N_DIMS]; N_DIMS], // Matrix of normalised probabilities
    W: [[f32; N_DIMS]; N_DIMS], // Our single layer of neurons
    nll: f32,                   // The normalised, negative log likelihood of the model
    training: Dataset,
    testing: Dataset,
    xs: Vec<usize>, // Temporary
    ys: Vec<usize>, // Temporary
    codec: Codec,
}

impl<'a> Bigram<'a> {
    fn new(dataset: &'a Vec<Vec<char>>) -> Self {
        Self {
            dataset,
            P: [[0.0; N_DIMS]; N_DIMS],
            W: square_uniform_probability_matrix(),
            nll: 0.0,
            training: (Vec::new(), Vec::new()),
            testing: (Vec::new(), Vec::new()),
            xs: Vec::new(),
            ys: Vec::new(),
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

                self.xs.push(i_ch1);
                self.ys.push(i_ch2);
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

        // Our list of probabilities for each input
        let mut P = mat_mul(a, transpose(self.W));
        exponentiate(&mut P);
        normalize(&mut P);

        // Calculate the model's performance, i.e., calculating the normalised
        // negative log probability of the model.
        let mut log_likelihood = 0.0;
        let mut n = 0;
        let mut nlls = vec![];
        for i in 0..5 {
            let x = self.xs[i];
            let y = self.ys[i];
            let prob = P[i][y];
            println!("probability: {prob:.4}");
            let ll = prob.ln();
            let nll = ll.abs();
            println!("nll: {nll:.4}");
            nlls.push(nll);
        }

        let nll_sum = nlls.iter().sum::<f32>();
        println!("average nll: {:.4?}", nll_sum / nlls.len() as f32);
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
