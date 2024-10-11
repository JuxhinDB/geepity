use std::collections::{BTreeMap, BTreeSet};

const TINY_SHAKESPEARE: &str = include_str!("../char-rnn/data/tinyshakespeare/input.txt");

type Encoder = BTreeMap<char, u16>;
type Decoder = BTreeMap<u16, char>;

fn main() {
    let mut chars = BTreeSet::new();

    for c in TINY_SHAKESPEARE.chars() {
        chars.insert(c);
    }

    println!("Vocab size:\t{}", chars.len());
    println!("Vocabulary:\t{chars:?}");

    let mut encoder = BTreeMap::new();
    let mut decoder = BTreeMap::new();

    for (i, c) in chars.iter().enumerate() {
        encoder.insert(*c, i as u16);
        decoder.insert(i as u16, *c);
    }

    let data = encode_source(&encoder);

    println!("encoded data: {:?}", &data[..10]);
}

// This would be a translation from:
// torch.tensor(encode(text), dtype=torch.long)
fn encode_source(encoder: &Encoder) -> Vec<u16> {
    TINY_SHAKESPEARE
        .chars()
        .map(|c| *encoder.get(&c).expect("couldn't find key in vocab"))
        .collect()
}
