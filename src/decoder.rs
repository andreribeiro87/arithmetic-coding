//! The [`Decoder`] half of the arithmetic coding library.

use std::{io, ops::Range};

use bitstream_io::BitRead;

#[cfg(debug_assertions)]
use crate::common::assert_precision_sufficient;
use crate::{
    BitStore, Model,
    common::{self},
};

// this algorithm is derived from this article - https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html

/// An arithmetic decoder
///
/// An arithmetic decoder converts a stream of bytes into a stream of some
/// output symbol, using a predictive [`Model`].
#[derive(Debug)]
pub struct Decoder<M, R>
where
    M: Model,
    R: BitRead,
{
    model: M,
    state: State<M::B, R>,
    idx_counter: usize,
}

impl<M, R> Decoder<M, R>
where
    M: Model,
    R: BitRead,
{
    /// Construct a new [`Decoder`]
    ///
    /// The 'precision' of the encoder is maximised, based on the number of bits
    /// needed to represent the [`Model::denominator`]. 'precision' bits is
    /// equal to [`u32::BITS`] - [`Model::denominator`] bits.
    ///
    /// # Panics
    ///
    /// The calculation of the number of bits used for 'precision' is subject to
    /// the following constraints:
    ///
    /// - The total available bits is [`u32::BITS`]
    /// - The precision must use at least 2 more bits than that needed to
    ///   represent [`Model::denominator`]
    ///
    /// If these constraints cannot be satisfied this method will panic in debug
    /// builds
    ///
    /// # Errors
    ///
    /// This method can fail if the underlying [`BitRead`] cannot be read from
    /// during initialisation.
    pub fn new(mut model: M, input: R) -> io::Result<Self> {
        let frequency_bits = model.max_denominator().log2() + 1;
        let precision = M::B::BITS - frequency_bits;

        Self::with_precision(model, input, precision)
    }

    /// Construct a new [`Decoder`] with a custom precision
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
    ///
    /// # Errors
    ///
    /// This method can fail if the underlying [`BitRead`] cannot be read from
    /// during initialisation.
    pub fn with_precision(model: M, input: R, precision: u32) -> io::Result<Self> {
        let state = State::new(precision, input)?;
        Ok(Self::with_state(state, model))
    }

    /// Create a decoder from an existing [`State`] and [`Model`].
    ///
    /// This is useful for manually chaining a shared buffer through multiple
    /// decoders.
    pub fn with_state(state: State<M::B, R>, mut model: M) -> Self {
        #[cfg(debug_assertions)]
        assert_precision_sufficient::<M>(model.max_denominator(), state.state.precision);

        Self { model, state, idx_counter: 0 }
    }

    /// Return an iterator over the decoded symbols.
    ///
    /// The iterator will continue returning symbols until EOF is reached
    pub const fn decode_all(&'_ mut self) -> DecodeIter<'_, M, R> {
        DecodeIter { decoder: self }
    }

    /// Read the next symbol from the stream of bits
    ///
    /// This method will return `Ok(None)` when EOF is reached.
    ///
    /// # Errors
    ///
    /// This method can fail if the underlying [`BitRead`] cannot be read from.
    #[allow(clippy::missing_panics_doc)]
    pub fn decode(&mut self) -> io::Result<Option<M::Symbol>> {
        let denominator = self.model.denominator();
        debug_assert!(
            denominator <= self.model.max_denominator(),
            "denominator is greater than maximum!"
        );
        let value = self.state.value(denominator);
        let symbol = self.model.symbol(value);

        let p = self
            .model
            .probability(symbol.as_ref(), self.idx_counter)
            .expect("this should not be able to fail. Check the implementation of the model.");

        self.state.scale(p, denominator)?;
        self.model.update(symbol.as_ref());

        self.idx_counter += 1;

        Ok(symbol)
    }

    /// Reuse the internal state of the Decoder with a new model.
    ///
    /// Allows for chaining multiple sequences of symbols from a single stream
    /// of bits
    pub fn chain<X>(self, model: X) -> Decoder<X, R>
    where
        X: Model<B = M::B>,
    {
        Decoder::with_state(self.state, model)
    }

    /// Return the internal model and state of the decoder.
    pub fn into_inner(self) -> (M, State<M::B, R>) {
        (self.model, self.state)
    }
}

/// The iterator returned by the [`Model::decode_all`] method
#[allow(missing_debug_implementations)]
pub struct DecodeIter<'a, M, R>
where
    M: Model,
    R: BitRead,
{
    decoder: &'a mut Decoder<M, R>,
}

impl<M, R> Iterator for DecodeIter<'_, M, R>
where
    M: Model,
    R: BitRead,
{
    type Item = io::Result<M::Symbol>;

    fn next(&mut self) -> Option<Self::Item> {
        self.decoder.decode().transpose()
    }
}

/// A convenience struct which stores the internal state of an [`Decoder`].
#[derive(Debug)]
pub struct State<B, R>
where
    B: BitStore,
    R: BitRead,
{
    #[allow(clippy::struct_field_names)]
    state: common::State<B>,
    input: R,
    x: B,
    read_buf: u64,
    read_buf_len: u32,
    eof: bool,
}

impl<B, R> State<B, R>
where
    B: BitStore,
    R: BitRead,
{
    /// Create a new [`State`] from an input stream of bits with a given
    /// precision.
    ///
    /// Eagerly reads the initial `precision` bits from the input during
    /// construction.
    ///
    /// # Errors
    ///
    /// This method can fail if the underlying [`BitRead`] cannot be read from.
    pub fn new(precision: u32, input: R) -> io::Result<Self> {
        let state = common::State::new(precision);

        let mut s = Self {
            state,
            input,
            x: B::ZERO,
            read_buf: 0,
            read_buf_len: 0,
            eof: false,
        };

        s.fill()?;
        Ok(s)
    }

    /// Pull a single bit from the local read buffer, refilling from the
    /// underlying reader when empty. Returns `false` on EOF.
    #[inline]
    fn pull_bit(&mut self) -> io::Result<bool> {
        if self.read_buf_len == 0 {
            if self.eof {
                return Ok(false);
            }
            self.refill()?;
            if self.read_buf_len == 0 {
                self.eof = true;
                return Ok(false);
            }
        }
        self.read_buf_len -= 1;
        Ok((self.read_buf >> self.read_buf_len) & 1 != 0)
    }

    /// Refill the local read buffer by reading whole bytes from the input.
    fn refill(&mut self) -> io::Result<()> {
        while self.read_buf_len <= 56 {
            match self.input.read::<8, u8>() {
                Ok(byte) => {
                    self.read_buf = (self.read_buf << 8) | u64::from(byte);
                    self.read_buf_len += 8;
                }
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    self.eof = true;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    #[inline]
    fn normalise(&mut self) -> io::Result<()> {
        let half = self.state.half;
        let quarter = self.state.quarter;
        let three_quarter = self.state.three_quarter;

        loop {
            if self.state.high < half {
                self.state.high = (self.state.high << 1) + B::ONE;
                self.state.low <<= 1;
                self.x <<= 1;
            } else if self.state.low >= half {
                self.state.low = (self.state.low - half) << 1;
                self.state.high = ((self.state.high - half) << 1) + B::ONE;
                self.x = (self.x - half) << 1;
            } else {
                break;
            }

            if self.pull_bit()? {
                self.x += B::ONE;
            }
        }

        while self.state.low >= quarter && self.state.high < three_quarter {
            self.state.low = (self.state.low - quarter) << 1;
            self.state.high = ((self.state.high - quarter) << 1) + B::ONE;
            self.x = (self.x - quarter) << 1;

            if self.pull_bit()? {
                self.x += B::ONE;
            }
        }

        Ok(())
    }

    #[inline]
    fn scale(&mut self, p: Range<B>, denominator: B) -> io::Result<()> {
        self.state.scale(p, denominator);
        self.normalise()
    }

    #[inline]
    fn value(&self, denominator: B) -> B {
        let range = self.state.high - self.state.low + B::ONE;
        ((self.x - self.state.low + B::ONE) * denominator - B::ONE) / range
    }

    fn fill(&mut self) -> io::Result<()> {
        for _ in 0..self.state.precision {
            self.x <<= 1;
            if self.pull_bit()? {
                self.x += B::ONE;
            }
        }
        Ok(())
    }
}
