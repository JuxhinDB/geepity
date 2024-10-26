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

/// Generates an [N, N] matrix with values sampled from the uniform distribution.
pub(crate) fn square_uniform_probability_matrix<const N: usize>() -> [[f32; N]; N] {
    let mut rng = rand::thread_rng();

    // NOTE(juxhin): This distribution isn't exactly right.. values can be
    // greater or smaller than 1.0 and -1.0 respectively.
    let distribution = Uniform::new_inclusive(-1.0, 1.0);

    // FIXME(juxhin): We'd likely want to initialise these directly
    let mut matrix = [[0.0f32; N]; N];

    for (i, _) in matrix.into_iter().enumerate() {
        // NOTE(juxhin): Explore unsafe alternative here to cast Vec<f32> to
        // [f32; N] as we are able to guarantee the bounds
        //
        // This not only fetches a sample, but exponentiates it and normalises
        // it. The benefit here is that we already know the length/sum as `N`, 
        // and therefore do not need to compute the length.
        let sample: Vec<f32> = distribution.sample_iter(&mut rng).take(N).collect();
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
    a: [[f32; D2]; D1],
    b: [[f32; D4]; D3],
) -> [[f32; D4]; D1] {
    debug_assert_eq!(D2, D3);
    let mut result = [[0.0f32; D4]; D1];

    for i in 0..D1 {
        for j in 0..D4 {
            let col = b.iter().map(|r| r[j]);
            result[i][j] = a[i].iter().zip(col).map(|(a, b)| a * b).sum();
        }
    }

    result
}

pub(crate) fn exponentiate<const D1: usize, const D2: usize>(a: &mut [[f32; D2]; D1]) {
    for i in 0..D1 {
        a[i] = vec_to_bounded_slice::<f32, D2>(a[i].iter().map(|v| v.exp()).collect());
    }
}

pub(crate) fn normalize<const D1: usize, const D2: usize>(a: &mut [[f32; D2]; D1]) {
    for i in 0..D1 {
        let row_sum = a[i].iter().sum::<f32>();
        a[i] = vec_to_bounded_slice(a[i].iter().map(|v| v / row_sum).collect())
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let a = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];

        let b = [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0]
        ];

        assert_eq!(transpose(a), b);
    }

    #[test]
    fn test_matmul() {
        let a = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ];

        let b = transpose(a);

        let result: [[f32; 2]; 2] = [
            [14.0, 32.0],
            [32.0, 77.0]
        ];

        assert_eq!(mat_mul::<2, 3, 3, 2>(a, b), result);
    }
}