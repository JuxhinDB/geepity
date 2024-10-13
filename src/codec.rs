use std::collections::{BTreeMap, BTreeSet};

pub(crate) const CONTROL: char = '\u{0000}';
pub(crate) const DOT_CONTROL: char = '.';

pub(crate) type Encoder = BTreeMap<char, u16>;
pub(crate) type Decoder = BTreeMap<u16, char>;

pub(crate) struct Codec {
    encoder: Encoder,
    decoder: Decoder,
}

impl Codec {
    pub fn new(source: &Vec<Vec<char>>) -> Self {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();

        let vocabulary = BTreeSet::from_iter(source.iter().flatten());

        for (i, c) in vocabulary.into_iter().enumerate() {
            encoder.insert(*c, (i + 1) as u16);
            decoder.insert((i + 1) as u16, *c);
        }

        encoder.insert(DOT_CONTROL, 0);
        decoder.insert(0, DOT_CONTROL);

        Self { encoder, decoder }
    }

    pub fn encode(&self, c: &char) -> usize {
        // FIXME(jdb): remove unwrap
        (*self.encoder.get(c).unwrap()).into()
    }

    pub fn decode(&self, i: u16) -> char {
        // FIXME(jdb): Remove unwrap
        *self.decoder.get(&i).unwrap()
    }
}
