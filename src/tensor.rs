pub use half::{bf16, f16};
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct TensorView<'a, T = f32>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub data: Cow<'a, [T]>,
    pub shape: Cow<'a, [usize]>,
}

pub type TensorViewF32<'a> = TensorView<'a, f32>;
pub type TensorViewI8<'a> = TensorView<'a, i8>;
pub type TensorViewU8<'a> = TensorView<'a, u8>;
pub type TensorViewI32<'a> = TensorView<'a, i32>;
pub type TensorViewI64<'a> = TensorView<'a, i64>;
pub type TensorViewF16<'a> = TensorView<'a, f16>;
pub type TensorViewBF16<'a> = TensorView<'a, bf16>;

impl<'a, T> TensorView<'a, T>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub fn new(data: &'a [T], shape: &'a [usize]) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Borrowed(data),
            shape: Cow::Borrowed(shape),
        }
    }

    pub fn from_owned(data: Vec<T>, shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Owned(data),
            shape: Cow::Owned(shape),
        }
    }

    pub fn to_owned(&self) -> TensorView<'static, T> {
        TensorView::from_owned(self.data.to_vec(), self.shape.to_vec())
    }

    pub fn empty() -> Self {
        Self {
            data: Cow::Borrowed(&[]),
            shape: Cow::Borrowed(&[]),
        }
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    pub fn from_slice(data: &'a [T], shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "Data length mismatch");
        Self {
            data: Cow::Borrowed(data),
            shape: Cow::Owned(shape),
        }
    }

    /// # Safety
    /// This function is unsafe because it bypasses the lifetime system to create a new `TensorView`
    /// that might outlive the data it points to.
    pub unsafe fn detach<'b>(&self) -> TensorView<'b, T> {
        unsafe {
            let slice = std::slice::from_raw_parts(self.data.as_ptr(), self.data.len());
            let shape_slice = std::slice::from_raw_parts(self.shape.as_ptr(), self.shape.len());
            TensorView {
                data: Cow::Borrowed(slice),
                shape: Cow::Borrowed(shape_slice),
            }
        }
    }
}

impl<'a> TensorView<'a, f32> {
    /// # Safety
    /// Reinterprets this TensorView<f32> as TensorView<u8>. This is only safe when the
    /// f32 values represent u8 data (e.g., from quantization operations).
    pub unsafe fn reinterpret_as_u8(&self) -> TensorView<'a, u8> {
        // The f32 vector actually contains u8 values stored as f32
        // We need to convert them back
        let u8_vec: Vec<u8> = self.data.iter().map(|&x| x as u8).collect();
        TensorView::from_owned(u8_vec, self.shape.to_vec())
    }
}

pub trait IntoLogits<'a, T>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn into_logits(self) -> TensorView<'a, T>;
}

impl<'a, T> IntoLogits<'a, T> for TensorView<'a, T>
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn into_logits(self) -> TensorView<'a, T> {
        self
    }
}

impl<'a, T, U> IntoLogits<'a, T> for (TensorView<'a, T>, U)
where
    T: Clone + std::fmt::Debug,
    [T]: ToOwned<Owned = Vec<T>>,
{
    fn into_logits(self) -> TensorView<'a, T> {
        self.0
    }
}
