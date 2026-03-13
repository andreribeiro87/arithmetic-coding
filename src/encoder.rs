//! The [`Encoder`] half of the arithmetic coding library.

use std::{io, ops::Range};

use bitstream_io::BitWrite;

#[cfg(debug_assertions)]
use crate::common::assert_precision_sufficient;
use crate::{
    BitStore, Error, Model,
    common::{self},
};

// this algorithm is derived from this article - https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html

/// An arithmetic encoder
///
/// An arithmetic decoder converts a stream of symbols into a stream of bits,
/// using a predictive [`Model`].
#[derive(Debug)]
pub struct Encoder<M, W>
where
    M: Model,
    W: BitWrite,
{
    model: M,
    state: State<M::B, W>,
}

impl<M, W> Encoder<M, W>
where
    M: Model,
    W: BitWrite,
{
    /// Construct a new [`Encoder`].
    ///
    /// The 'precision' of the encoder is maximised, based on the number of bits
    /// needed to represent the [`Model::denominator`]. 'precision' bits is
    /// equal to [`BitStore::BITS`] - [`Model::denominator`] bits. If you need
    /// to set the precision manually, use [`Encoder::with_precision`].
    ///
    /// # Panics
    ///
    /// The calculation of the number of bits used for 'precision' is subject to
    /// the following constraints:
    ///
    /// - The total available bits is [`BitStore::BITS`]
    /// - The precision must use at least 2 more bits than that needed to
    ///   represent [`Model::denominator`]
    ///
    /// If these constraints cannot be satisfied this method will panic in debug
    /// builds
    pub fn new(mut model: M, bitwriter: W) -> Self {
        let frequency_bits = model.max_denominator().log2() + 1;
        let precision = M::B::BITS - frequency_bits;
        Self::with_precision(model, bitwriter, precision)
    }

    /// Construct a new [`Encoder`] with a custom precision.
    ///
    /// # Panics
    ///
    /// The calculation of the number of bits used for 'precision' is subject to
    /// the following constraints:
    ///
    /// - The total available bits is [`BitStore::BITS`]
    /// - The precision must use at least 2 more bits than that needed to
    ///   represent [`Model::denominator`]
    ///
    /// If these constraints cannot be satisfied this method will panic in debug
    /// builds
    pub fn with_precision(model: M, bitwriter: W, precision: u32) -> Self {
        let state = State::new(precision, bitwriter);
        Self::with_state(state, model)
    }

    /// Create an encoder from an existing [`State`].
    ///
    /// This is useful for manually chaining a shared buffer through multiple
    /// encoders.
    pub fn with_state(state: State<M::B, W>, mut model: M) -> Self {
        #[cfg(debug_assertions)]
        assert_precision_sufficient::<M>(model.max_denominator(), state.state.precision);
        Self { model, state }
    }

    /// Encode a stream of symbols into the provided output.
    ///
    /// This method will encode all the symbols in the iterator, followed by EOF
    /// (`None`), and then call [`Encoder::flush`].
    ///
    /// # Errors
    ///
    /// This method can fail if the underlying [`BitWrite`] cannot be written
    /// to.
    pub fn encode_all(
        &mut self,
        symbols: impl IntoIterator<Item = M::Symbol>,
    ) -> Result<(), Error<M::ValueError>> {
        for (index, symbol) in symbols.into_iter().enumerate() {
            self.encode(Some(&symbol), index)?;
        }
        // self.encode(None)?;
        self.flush()?;
        Ok(())
    }

    /// Encode a symbol into the provided output.
    ///
    /// When you finish encoding symbols, you must manually encode an EOF symbol
    /// by calling [`Encoder::encode`] with `None`.
    ///
    /// The internal buffer must be manually flushed using [`Encoder::flush`].
    ///
    /// # Errors
    ///
    /// This method can fail if the underlying [`BitWrite`] cannot be written
    /// to.
    pub fn encode(&mut self, symbol: Option<&M::Symbol>, index: usize) -> Result<(), Error<M::ValueError>> {
        let p = self.model.probability(symbol, index).map_err(Error::ValueError)?;
        let denominator = self.model.denominator();
        debug_assert!(
            denominator <= self.model.max_denominator(),
            "denominator is greater than maximum!"
        );

        self.state.scale(p, denominator)?;
        self.model.update(symbol);

        Ok(())
    }

    /// Flush any pending bits from the buffer
    ///
    /// This method must be called when you finish writing symbols to a stream
    /// of bits. This is called automatically when you use
    /// [`Encoder::encode_all`].
    ///
    /// # Errors
    ///
    /// This method can fail if the underlying [`BitWrite`] cannot be written
    /// to.
    pub fn flush(&mut self) -> io::Result<()> {
        self.state.flush()
    }

    /// Return the internal model and state of the encoder.
    pub fn into_inner(self) -> (M, State<M::B, W>) {
        (self.model, self.state)
    }

    /// Reuse the internal state of the Encoder with a new model.
    ///
    /// Allows for chaining multiple sequences of symbols into a single stream
    /// of bits
    pub fn chain<X>(self, model: X) -> Encoder<X, W>
    where
        X: Model<B = M::B>,
    {
        Encoder::with_state(self.state, model)
    }

    /// Return the alphabet of the model.
    pub fn alphabet(&mut self) -> Vec<M::Symbol> {
        self.model.alphabet()
    }
}

/// A convenience struct which stores the internal state of an [`Encoder`].
#[derive(Debug)]
pub struct State<B, W>
where
    B: BitStore,
    W: BitWrite,
{
    #[allow(clippy::struct_field_names)]
    state: common::State<B>,
    pending: u32,
    output: W,
    /// Local bit-packing register. Bits are accumulated LSB-first:
    /// the oldest pushed bit sits at position `write_buf_len - 1`,
    /// the newest at position 0. This mirrors the behaviour of
    /// `BitWrite::write_var` which writes the N least-significant bits
    /// MSB-first, giving the correct bit order on the wire.
    write_buf: u64,
    /// Number of valid bits currently held in `write_buf` (always < 64;
    /// the buffer is flushed to `output` the moment it reaches 64).
    write_buf_len: u32,
}

impl<B, W> State<B, W>
where
    B: BitStore,
    W: BitWrite,
{
    /// Manually construct a [`State`].
    ///
    /// Normally this would be done automatically using the [`Encoder::new`]
    /// method.
    pub fn new(precision: u32, output: W) -> Self {
        Self {
            state: common::State::new(precision),
            pending: 0,
            output,
            write_buf: 0,
            write_buf_len: 0,
        }
    }

    /// Write the full 64-bit buffer to the underlying writer and reset it.
    #[inline]
    fn flush_full_buf(&mut self) -> io::Result<()> {
        debug_assert_eq!(self.write_buf_len, 64);
        self.output.write::<64, u64>(self.write_buf)?;
        self.write_buf = 0;
        self.write_buf_len = 0;
        Ok(())
    }

    /// Push a single bit into the local buffer.
    ///
    /// Flushes a full 64-bit word to `output` whenever the buffer fills up,
    /// keeping `write_buf_len` strictly in `0..64` on return.
    #[inline]
    fn push_bit(&mut self, bit: bool) -> io::Result<()> {
        self.write_buf = (self.write_buf << 1) | u64::from(bit);
        self.write_buf_len += 1;
        if self.write_buf_len == 64 {
            self.flush_full_buf()?;
        }
        Ok(())
    }

    /// Push `count` identical copies of `bit` into the local buffer,
    /// flushing whole 64-bit words to `output` along the way.
    fn push_bits_repeated(&mut self, bit: bool, mut count: u32) -> io::Result<()> {
        if count == 0 {
            return Ok(());
        }

        // ── Step 1: fill any partial tail in the current buffer ────────────
        if self.write_buf_len > 0 {
            let space = 64 - self.write_buf_len;
            let fill = count.min(space);
            // write_buf_len >= 1  ⟹  space <= 63  ⟹  fill <= 63,
            // so `1u64 << fill` never overflows.
            let mask: u64 = if bit { (1u64 << fill) - 1 } else { 0 };
            self.write_buf = (self.write_buf << fill) | mask;
            self.write_buf_len += fill;
            count -= fill;

            if self.write_buf_len == 64 {
                self.output.write::<64, u64>(self.write_buf)?;
                self.write_buf = 0;
                self.write_buf_len = 0;
            } else {
                // Buffer still has room; all bits have been consumed.
                return Ok(());
            }
        }

        // ── Step 2: buffer is empty — blast out full 64-bit words ──────────
        let word: u64 = if bit { u64::MAX } else { 0 };
        while count >= 64 {
            self.output.write::<64, u64>(word)?;
            count -= 64;
        }

        // ── Step 3: stash the remaining < 64 bits in the (empty) buffer ────
        if count > 0 {
            // count < 64 ⟹ `1u64 << count` is safe.
            let mask: u64 = if bit { (1u64 << count) - 1 } else { 0 };
            self.write_buf = mask;
            self.write_buf_len = count;
        }

        Ok(())
    }

    /// Flush any bits that are still sitting in the local buffer out to
    /// `output`.  After this call `write_buf_len` is 0.
    fn drain_write_buf(&mut self) -> io::Result<()> {
        if self.write_buf_len > 0 {
            // write_buf_len is in 1..=63 here (the buffer is flushed the
            // moment it reaches 64 inside push_bit / push_bits_repeated).
            self.output
                .write_var::<u64>(self.write_buf_len, self.write_buf)?;
            self.write_buf = 0;
            self.write_buf_len = 0;
        }
        Ok(())
    }

    #[inline]
    fn scale(&mut self, p: Range<B>, denominator: B) -> io::Result<()> {
        self.state.scale(p, denominator);
        self.normalise()
    }

    #[inline]
    fn normalise(&mut self) -> io::Result<()> {
        let half = self.state.half;
        let quarter = self.state.quarter;
        let three_quarter = self.state.three_quarter;

        loop {
            if self.state.high < half {
                self.emit(false)?;
                self.state.high = (self.state.high << 1) + B::ONE;
                self.state.low <<= 1;
            } else if self.state.low >= half {
                self.emit(true)?;
                self.state.low = (self.state.low - half) << 1;
                self.state.high = ((self.state.high - half) << 1) + B::ONE;
            } else {
                break;
            }
        }

        while self.state.low >= quarter && self.state.high < three_quarter {
            self.pending += 1;
            self.state.low = (self.state.low - quarter) << 1;
            self.state.high = ((self.state.high - quarter) << 1) + B::ONE;
        }

        Ok(())
    }

    /// Emit a resolved bit followed by all accumulated straddle bits.
    ///
    /// All output goes through the local u64 buffer rather than directly
    /// to `self.output`, minimising the number of `BitWrite` calls.
    #[inline]
    fn emit(&mut self, bit: bool) -> io::Result<()> {
        self.push_bit(bit)?;
        let n = self.pending;
        if n > 0 {
            self.push_bits_repeated(!bit, n)?;
            self.pending = 0;
        }
        Ok(())
    }

    /// Flush the internal buffer and write all remaining bits to the output.
    /// This method MUST be called when you finish writing symbols to ensure
    /// they are fully written to the output.
    ///
    /// # Errors
    ///
    /// This method can fail if the output cannot be written to
    pub fn flush(&mut self) -> io::Result<()> {
        self.pending += 1;
        if self.state.low <= self.state.quarter {
            self.emit(false)?;
        } else {
            self.emit(true)?;
        }
        self.drain_write_buf()?;
        Ok(())
    }
}
