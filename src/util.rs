pub(crate) fn one_hot<const N: usize>(v: usize) -> Vec<f32> {
    let mut slice = vec![0.0; N];
    slice[v] = 1.0f32;
    slice
}

pub(crate) fn vec_to_bounded_slice<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into().unwrap_or_else(|v: Vec<T>| {
        panic!("Expected a Vec of length {} but it was {}", N, v.len())
    })
}

pub(crate) fn square_weighted_matrix<const N: usize>() -> Vec<Vec<f32>> {
    todo!()    
}