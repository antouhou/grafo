/// Sums `f64` values while compensating for rounding error.
#[derive(Clone, Copy, Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    error: f64,
}

impl CompensatedSum {
    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        let sum = self.sum + value;
        let rounded_value = sum - self.sum;
        self.error += (self.sum - (sum - rounded_value)) + (value - rounded_value);
        self.sum = sum;
    }

    #[inline]
    pub(crate) fn add_sum(&mut self, other: Self) {
        self.add(other.sum);
        self.add(other.error);
    }

    #[inline]
    pub(crate) fn total(&self) -> f64 {
        self.sum + self.error
    }
}
