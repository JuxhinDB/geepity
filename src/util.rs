use rand::distributions::{Distribution, Uniform};

pub(crate) fn one_hot<const D: usize>(index: usize) -> Vec<f32> {
    let mut vec = vec![0.0; D];
    vec[index] = 1.0;
    vec
}

/// Generates an [n x n] matrix with values sampled from the uniform distribution.
pub(crate) fn square_uniform_probability_matrix<const D: usize>() -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let distribution = Uniform::new_inclusive(-0.01, 0.01);

    let mut matrix = Vec::with_capacity(D);
    for _ in 0..D {
        let row: Vec<f32> = distribution.sample_iter(&mut rng).take(D).collect();
        matrix.push(row);
    }

    matrix
}

/// Transposes a matrix represented as a vector of vectors.
pub(crate) fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(!matrix.is_empty());
    let nrows = matrix.len();
    let ncols = matrix[0].len();
    let mut transposed = vec![vec![0.0; nrows]; ncols];

    for i in 0..nrows {
        for j in 0..ncols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}

/// Performs matrix multiplication between two matrices.
pub(crate) fn mat_mul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let a_rows = a.len();
    let a_cols = a[0].len();
    let b_rows = b.len();
    let b_cols = b[0].len();
    assert_eq!(a_cols, b_rows);

    let mut result = vec![vec![0.0; b_cols]; a_rows];
    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

pub(crate) fn mat_mul_single(a: &[f32], b: &[Vec<f32>]) -> Vec<f32> {
    let a_len = a.len();
    let b_rows = b.len();
    let b_cols = b[0].len();
    assert_eq!(a_len, b_rows);

    let mut result = vec![0.0; b_cols];
    for j in 0..b_cols {
        for i in 0..a_len {
            result[j] += a[i] * b[i][j];
        }
    }
    result
}

pub(crate) fn exponentiate_single(row: &mut [f32]) {
    for v in row.iter_mut() {
        *v = v.exp();
    }
}

pub(crate) fn normalize_single(row: &mut [f32]) {
    let sum: f32 = row.iter().sum();
    for v in row.iter_mut() {
        *v /= sum;
    }
}

pub(crate) fn exponentiate(a: &mut [Vec<f32>]) {
    for row in a.iter_mut() {
        exponentiate_single(row);
    }
}

pub(crate) fn normalize(a: &mut [Vec<f32>]) {
    for row in a.iter_mut() {
        normalize_single(row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let expected = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];

        assert_eq!(transpose(&a), expected);
    }

    #[test]
    fn test_matmul() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = transpose(&a);

        let expected = vec![vec![14.0, 32.0], vec![32.0, 77.0]];

        assert_eq!(mat_mul(&a, &b), expected);
    }

    #[test]
    fn test_one_hot() {
        let index = 2;
        let expected = vec![0.0, 0.0, 1.0, 0.0, 0.0];

        assert_eq!(one_hot::<5>(index), expected);
    }

    #[test]
    fn test_exponentiate_and_normalize_single() {
        let mut row = vec![1.0, 2.0, 3.0];
        exponentiate_single(&mut row);
        normalize_single(&mut row);

        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
