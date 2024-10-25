use rand::distributions::{Distribution, Uniform};

pub(crate) fn one_hot<const N: usize>(v: usize) -> Vec<f32> {
    let mut slice = vec![0.0; N];
    slice[v] = 1.0f32;
    slice
}

pub(crate) fn vec_to_bounded_slice<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

/// Generates an [N, N] matrix with values sampled from the uniform distribution
///
/// In our case this is meant to be equivalent to `torch.rand`
pub(crate) fn square_uniform_matrix<const N: usize>() -> [[f32; N]; N] {
    let mut rng = rand::thread_rng();

    // NOTE(juxhin): This distribution isn't exactly right.. values can be
    // greater or smaller than 1.0 and -1.0 respectively.
    let distribution = Uniform::new_inclusive(-1.0, 1.0);

    // FIXME(juxhin): We'd likely want to initialise these directly
    let mut matrix = [[0.0f32; N]; N];

    for (i, _) in matrix.into_iter().enumerate() {
        // NOTE(juxhin): Explore unsafe alternative here to cast Vec<f32> to
        // [f32; N] as we are able to guarantee the bounds
        let sample = distribution.sample_iter(&mut rng).take(N).collect();
        matrix[i] = vec_to_bounded_slice::<f32, N>(sample);
    }

    matrix
}

// Taken and adapted from:
// https://stackoverflow.com/questions/64498617/how-to-transpose-a-vector-of-vectors-in-rust
pub(crate) fn transpose<const D1: usize, const D2: usize>(
    original: [[f32; D2]; D1],
) -> [[f32; D1]; D2] {
    assert!(!original.is_empty());
    let mut transposed = (0..original[0].len())
        .map(|_| [0.0; D1])
        .collect::<Vec<_>>();

    for (i, original_row) in original.into_iter().enumerate() {
        for (item, transposed_row) in original_row.into_iter().zip(&mut transposed) {
            transposed_row[i] = item;
        }
    }

    vec_to_bounded_slice(transposed)
}

/// Pass in the shapes of the two matrices, the output is the shape of the `D1`
/// and `D4`. We can likely simplify this quite a lot. Also this is not
/// designed to be performant in any way (_yet_).
pub(crate) fn mat_mul<const D1: usize, const D2: usize, const D3: usize, const D4: usize>(
    a: [[f32; D1]; D2],
    b: [[f32; D3]; D4],
) -> [[f32; D4]; D2] {
    debug_assert_eq!(D1, D3);
    let mut result = [[0.0f32; D4]; D2];

    for i in 0..D2 {
        for j in 0..D4 {
            result[i][j] = a[i].iter().zip(b[j].iter()).map(|(a, b)| a * b).sum();
        }
    }

    result
}
