use std::ops::Range;

use arithmetic_coding_core::BitStore;

use crate::Model;

#[derive(Debug)]
pub struct State<B: BitStore> {
    pub precision: u32,
    pub low: B,
    pub high: B,
    pub half: B,
    pub quarter: B,
    pub three_quarter: B,
}

impl<B> State<B>
where
    B: BitStore,
{
    pub fn new(precision: u32) -> Self {
        let half = B::ONE << (precision - 1);
        let quarter = B::ONE << (precision - 2);
        let three_quarter = half + quarter;

        Self {
            precision,
            low: B::ZERO,
            high: (B::ONE << precision) - B::ONE,
            half,
            quarter,
            three_quarter,
        }
    }

    #[inline]
    pub fn scale(&mut self, p: Range<B>, denominator: B) {
        let range = self.high - self.low + B::ONE;

        if denominator.is_power_of_two() {
            let shift = denominator.trailing_zeros();
            self.high = self.low + ((range * p.end) >> shift) - B::ONE;
            self.low += (range * p.start) >> shift;
        } else {
            self.high = self.low + (range * p.end) / denominator - B::ONE;
            self.low += (range * p.start) / denominator;
        }
    }
}

pub fn assert_precision_sufficient<M: Model>(max_denominator: M::B, precision: u32) {
    let frequency_bits = max_denominator.log2() + 1;
    assert!(
        (precision >= (frequency_bits + 2)),
        "not enough bits of precision to prevent overflow/underflow",
    );
    assert!(
        (frequency_bits + precision) <= M::B::BITS,
        "not enough bits in BitStore to support the required precision",
    );
}
