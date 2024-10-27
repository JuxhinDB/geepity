use rand::{distributions::WeightedIndex, prelude::*};

use crate::{
    codec::{Codec, DOT_CONTROL},
    util::{exponentiate_single, mat_mul, mat_mul_single, normalize_single, one_hot, square_uniform_probability_matrix, transpose},
};

const NAMES_DATASET: &str = include_str!("../makemore/names.txt");
const N_DIMS: usize = 28; // Length of our vocabulary
const LEARNING_RATE: f32 = 1.0;

type Dataset = (Vec<Vec<f32>>, Vec<usize>); // Maps to inputs and target indices

struct Bigram<'a> {
    dataset: &'a Vec<Vec<char>>,
    P: Vec<Vec<f32>>,          // Matrix of normalized probabilities
    W: Vec<Vec<f32>>,          // Our single layer of neurons
    dW: Vec<Vec<f32>>,         // Derivative of W
    nll: f32,                  // The normalized negative log likelihood of the model
    training: Dataset,
    xs: Vec<usize>,            // Input indices
    ys: Vec<usize>,            // Target indices
    codec: Codec,
}

impl<'a> Bigram<'a> {
    fn new(dataset: &'a Vec<Vec<char>>) -> Self {
        Self {
            dataset,
            P: Vec::new(),
            W: square_uniform_probability_matrix::<N_DIMS>(),
            dW: vec![vec![0.0; N_DIMS]; N_DIMS],
            nll: 0.0,
            training: (Vec::new(), Vec::new()),
            xs: Vec::new(),
            ys: Vec::new(),
            codec: Codec::new(dataset),
        }
    }

    fn compute(&mut self) {
        for name in self.dataset.iter() {
            let control = [DOT_CONTROL];

            let chars = control.iter().chain(name.iter()).chain(control.iter());

            for (ch1, ch2) in chars.clone().zip(chars.skip(1)) {
                let i_ch1 = self.codec.encode(ch1);
                let i_ch2 = self.codec.encode(ch2);

                self.training.0.push(one_hot::<N_DIMS>(i_ch1));
                self.training.1.push(i_ch2);

                self.xs.push(i_ch1);
                self.ys.push(i_ch2);
            }
        }
    }

    fn forward(&mut self) {
        let a = &self.training.0;

        // Calculate logits: a @ W.T
        self.P = mat_mul(a, &transpose(&self.W));

        // Apply exponentiation and normalization to get probabilities
        for row in &mut self.P {
            exponentiate_single(row);
            normalize_single(row);
        }

        // Calculate loss (negative log likelihood)
        self.nll = -self
            .ys
            .iter()
            .enumerate()
            .map(|(i, &y)| self.P[i][y].ln())
            .sum::<f32>()
            / self.ys.len() as f32;
    }

    fn backward(&mut self) {
        let batch_size = self.training.0.len();

        // Reset gradients
        self.dW = vec![vec![0.0; N_DIMS]; N_DIMS];

        for i in 0..batch_size {
            let x = &self.training.0[i];
            let y = self.ys[i];

            let mut logit_gradient = self.P[i].clone();
            logit_gradient[y] -= 1.0;

            for k in 0..N_DIMS {
                for j in 0..N_DIMS {
                    self.dW[j][k] += x[j] * logit_gradient[k] / batch_size as f32;
                }
            }
        }

        // Update weights
        for i in 0..N_DIMS {
            for j in 0..N_DIMS {
                self.W[i][j] -= LEARNING_RATE * self.dW[i][j];
            }
        }
    }

    fn train(&mut self) {
        self.compute();

        for epoch in 0..100 {
            self.forward();
            self.backward();

            println!("Epoch {}, loss: {:?}", epoch, self.nll);
        }
    }

    fn sample(&self) -> Vec<char> {
        let mut out = vec![];
        let mut rng = rand::thread_rng();
        let mut idx = self.codec.encode(&DOT_CONTROL);

        loop {
            // Create a one-hot vector for the current character
            let x = one_hot::<N_DIMS>(idx);

            // Compute logits for the next character
            let logits = mat_mul_single(&x, &self.W);

            // Convert logits to probabilities
            let mut probs = logits.clone();
            exponentiate_single(&mut probs);
            normalize_single(&mut probs);

            // Sample the next character
            let dist = WeightedIndex::new(&probs).unwrap();
            idx = dist.sample(&mut rng);

            if idx == self.codec.encode(&DOT_CONTROL) {
                break;
            }

            out.push(self.codec.decode(idx as u16));
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train() {
        let dataset: Vec<Vec<char>> = NAMES_DATASET
            .split('\n')
            .map(|name| name.chars().collect())
            .collect();

        let mut bigram = Bigram::new(&dataset);
        bigram.train();

        let name_generated = bigram.sample();
        println!("Generated name: {:?}", name_generated);
    }
}
