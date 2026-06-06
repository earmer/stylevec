use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

fn char_ngrams(text: &str, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return Vec::new();
    }
    let mut grams = Vec::with_capacity(chars.len() - n + 1);
    for i in 0..=chars.len() - n {
        let mut s = String::with_capacity(n);
        for j in 0..n {
            s.push(chars[i + j]);
        }
        grams.push(s);
    }
    grams
}

fn jaccard_ngram_similarity(a: &str, b: &str, n: usize) -> f64 {
    let a_grams = char_ngrams(a, n);
    let b_grams = char_ngrams(b, n);

    if a_grams.is_empty() && b_grams.is_empty() {
        return 1.0;
    }

    let a_set: HashSet<String> = a_grams.into_iter().collect();
    let b_set: HashSet<String> = b_grams.into_iter().collect();
    let inter = a_set.intersection(&b_set).count();
    let union = a_set.union(&b_set).count();
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let rows = a_chars.len() + 1;
    let cols = b_chars.len() + 1;

    let mut dp = vec![vec![0usize; cols]; rows];
    for i in 0..rows {
        dp[i][0] = i;
    }
    for j in 0..cols {
        dp[0][j] = j;
    }

    for i in 1..rows {
        for j in 1..cols {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            let ins = dp[i][j - 1] + 1;
            let del = dp[i - 1][j] + 1;
            let sub = dp[i - 1][j - 1] + cost;
            dp[i][j] = ins.min(del).min(sub);
        }
    }
    dp[rows - 1][cols - 1]
}

fn normalized_levenshtein_similarity(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein_distance(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

fn average_similarity(a: &str, b: &str, n: usize) -> f64 {
    (jaccard_ngram_similarity(a, b, n) + normalized_levenshtein_similarity(a, b)) / 2.0
}

pub fn batch_avg_similarity_rs(pairs: &[(String, String)], n: usize) -> Vec<f64> {
    pairs
        .par_iter()
        .map(|(a, b)| average_similarity(a, b, n))
        .collect()
}

#[pyfunction]
fn batch_avg_similarity(py: Python<'_>, pairs: Vec<(String, String)>, n: usize) -> PyResult<Vec<f64>> {
    let out = py.detach(|| batch_avg_similarity_rs(&pairs, n));
    Ok(out)
}

#[pymodule]
fn simlar(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_avg_similarity, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg_similarity() {
        let pairs = vec![("abc".to_string(), "abc".to_string())];
        let scores = batch_avg_similarity_rs(&pairs, 2);
        assert_eq!(scores[0], 1.0);
    }
}
