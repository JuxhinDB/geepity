mod bigram;
mod codec;
mod util;

use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    sync::RwLock,
};

use codec::Codec;
use rand::{distributions::Uniform, rngs::ThreadRng, Rng};

const TINY_SHAKESPEARE: &str = include_str!("../char-rnn/data/tinyshakespeare/input.txt");

// How many independent tokens will we process in parallel
const BATCH_SIZE: usize = 4;

// How many tokens are in a single batch (i.e., the context window)
const BLOCK_SIZE: usize = 8;

type Encoder = BTreeMap<char, u16>;
type Decoder = BTreeMap<u16, char>;

type Tensor2D = Vec<Vec<u16>>;

fn main() {
    let chars = TINY_SHAKESPEARE.chars();
    let mut vocabulary = BTreeSet::new();

    for c in chars {
        vocabulary.insert(c);
    }

    println!("Vocab size:\t{}", vocabulary.len());
    println!("Vocabulary:\t{vocabulary:?}");

    // FIXME(jdb): Consolidate this mess
    let codec_source = &TINY_SHAKESPEARE
        .split("\n")
        .map(|sentence| sentence.chars().collect())
        .collect();

    let codec = Codec::new(codec_source);

    //let data = TINY_SHAKESPEARE;

    //println!("encoded data: {:?}", &data[..10]);

    //let split_at = (0.9 * data.len() as f32) as usize;
    //let training_data = &data[..split_at];
    //let validate_data = &data[split_at..];

    //println!("train_data: {:?}", &training_data[..=BLOCK_SIZE]);

    //let (x_batch, y_batch) = get_batch(training_data);
}

fn get_batch(dataset: &[u16]) -> (Tensor2D, Tensor2D) {
    let rng = rand::thread_rng();

    let distribution = Uniform::from(0..dataset.len() - BLOCK_SIZE);
    let indices: Vec<usize> = rng.sample_iter(distribution).take(BATCH_SIZE).collect();

    let mut x = vec![];
    let mut y = vec![];

    for index in indices {
        x.push(dataset[index..index + BLOCK_SIZE].to_vec());
        y.push(dataset[index + 1..=index + BLOCK_SIZE].to_vec());
    }

    (x, y)
}
