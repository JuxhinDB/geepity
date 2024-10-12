use std::collections::{BTreeMap, BTreeSet};

type Encoder = BTreeMap<char, u16>;
type Decoder = BTreeMap<u16, char>;

struct Codec<'a> {
    source: &'a Vec<char>,
}

impl<'a> Codec<'a> {
    fn build(&self) -> (Encoder, Decoder) {
        let mut encoder = Encoder::new();
        let mut decoder = Decoder::new();

        for (i, c) in self.source.iter().enumerate() {
            encoder.insert(*c, i as u16);
            decoder.insert(i as u16, *c);
        }

        (encoder, decoder)
    }
}
