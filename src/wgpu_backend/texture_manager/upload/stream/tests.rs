use super::{feed_texture_stream, TextureBatchWriter};
use crate::render_backend::texture_manager::{
    AlphaMode, TexturePixelFormat, TexturePixelLayout, TextureStorageFormat, TextureStreamError,
};
use crate::render_backend::PixelBufferError;
use crate::wgpu_backend::texture_manager::operations::TextureUploadOperations;
use crate::wgpu_backend::texture_manager::upload::{
    TextureUploadBatch, TEXTURE_UPLOAD_BATCH_BYTE_LIMIT,
};
use crate::wgpu_backend::texture_manager::{TextureManagerError, TextureTransferError};
use crate::wgpu_backend::WgpuBackendError;
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::future;
use futures::stream;
use futures::StreamExt;
use std::cell::RefCell;
use std::future::Future;
use std::ops::Range;
use std::pin::pin;
use std::rc::Rc;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Debug)]
struct SourceFailure;

struct RecordedTextureBatch {
    bytes: Vec<u8>,
    origin: (u32, u32),
    size: (u32, u32),
    bytes_per_row: u32,
}

struct LocalPixelChunk {
    pixels: Rc<[u8]>,
    range: Range<usize>,
}

impl AsRef<[u8]> for LocalPixelChunk {
    fn as_ref(&self) -> &[u8] {
        &self.pixels[self.range.clone()]
    }
}

#[derive(Default)]
struct RecordingTextureBatchWriter {
    batches: Rc<RefCell<Vec<RecordedTextureBatch>>>,
    resume: Option<oneshot::Receiver<()>>,
}

struct StreamWake(mpsc::Sender<()>);

impl TextureBatchWriter for RecordingTextureBatchWriter {
    async fn write(&mut self, batch: TextureUploadBatch<'_>) -> Result<(), WgpuBackendError> {
        if let Some(resume) = self.resume.take() {
            resume.await.map_err(|_| WgpuBackendError::DeviceLost)?;
        }
        self.batches.borrow_mut().push(RecordedTextureBatch {
            bytes: batch.bytes.to_vec(),
            origin: batch.origin,
            size: batch.size,
            bytes_per_row: batch.bytes_per_row,
        });
        Ok(())
    }
}

impl Wake for StreamWake {
    fn wake(self: Arc<Self>) {
        let _ = self.0.send(());
    }
}

fn wake_channel() -> (Waker, mpsc::Receiver<()>) {
    let (sender, receiver) = mpsc::channel();
    (Waker::from(Arc::new(StreamWake(sender))), receiver)
}

fn rgba_layout(size: (u32, u32)) -> TexturePixelLayout {
    TexturePixelLayout::rgba8_srgb(size, AlphaMode::PremultipliedInLinearSpace)
}

#[test]
fn local_sources_and_writers_preserve_fragmented_conversion_across_batches() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let size = (1024, 769);
    let pixels: Rc<[u8]> = (0..size.0 * size.1)
        .flat_map(|_| [3, 7, 11, 255])
        .collect::<Vec<_>>()
        .into();
    let byte_length = pixels.len();
    let source = stream::iter((0..byte_length).step_by(65_537).map(move |start| {
        Ok::<_, SourceFailure>(LocalPixelChunk {
            pixels: Rc::clone(&pixels),
            range: start..(start + 65_537).min(byte_length),
        })
    }));
    let batches = Rc::new(RefCell::new(Vec::new()));
    let (resume_sender, resume_receiver) = oneshot::channel();
    let mut writer = RecordingTextureBatchWriter {
        batches: Rc::clone(&batches),
        resume: Some(resume_receiver),
    };
    let layout = TexturePixelLayout {
        format: TexturePixelFormat::Bgra8,
        ..rgba_layout(size)
    };
    let (result, ()) = block_on(future::join(
        feed_texture_stream(
            &operation,
            TextureStorageFormat::Rgba8Srgb,
            layout,
            source,
            &mut writer,
        ),
        async move {
            resume_sender.send(()).unwrap();
        },
    ));
    assert!(result.is_ok());
    let batches = batches.borrow();
    assert_eq!(batches.len(), 4);
    for (index, batch) in batches.iter().enumerate() {
        let rows = if index == 3 { 1 } else { 256 };
        assert_eq!(batch.origin, (0, index as u32 * 256));
        assert_eq!(batch.size, (1024, rows));
        assert_eq!(batch.bytes_per_row, 4096);
        assert_eq!(batch.bytes.len(), rows as usize * 4096);
        assert!(batch
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == [11, 7, 3, 255]));
    }
}

#[test]
fn pending_borrowed_source_wakes_and_preserves_fragmented_pixels() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let pixels = [12, 34, 56, 255, 78, 90, 123, 255];
    let (resume_sender, resume_receiver) = oneshot::channel();
    let source = stream::iter([Ok::<_, SourceFailure>(&pixels[..3])]).chain(stream::once(async {
        resume_receiver.await.unwrap();
        Ok::<_, SourceFailure>(&pixels[3..])
    }));
    let mut writer = RecordingTextureBatchWriter::default();
    let (waker, wakes) = wake_channel();
    let mut context = Context::from_waker(&waker);
    {
        let mut upload = pin!(feed_texture_stream(
            &operation,
            TextureStorageFormat::Rgba8Srgb,
            rgba_layout((2, 1)),
            source,
            &mut writer,
        ));
        assert!(upload.as_mut().poll(&mut context).is_pending());
        resume_sender.send(()).unwrap();
        wakes.recv().unwrap();
        assert!(matches!(
            upload.as_mut().poll(&mut context),
            Poll::Ready(Ok(()))
        ));
    }
    let batches = writer.batches.borrow();
    assert_eq!(batches.len(), 1);
    assert_eq!(&batches[0].bytes[..pixels.len()], &pixels);
    assert_eq!(batches[0].bytes_per_row, 256);
}

#[test]
fn source_failure_after_a_submitted_band_preserves_the_source_error() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let band = vec![37; TEXTURE_UPLOAD_BATCH_BYTE_LIMIT];
    let failure = Rc::new(());
    let source = stream::iter([Ok(band.as_slice()), Err(Rc::clone(&failure))]);
    let mut writer = RecordingTextureBatchWriter::default();
    let result = block_on(feed_texture_stream(
        &operation,
        TextureStorageFormat::Rgba8Srgb,
        rgba_layout((1024, 257)),
        source,
        &mut writer,
    ));
    let Err(TextureStreamError::Source(returned_failure)) = result else {
        panic!("expected the original local source failure");
    };
    assert!(Rc::ptr_eq(&returned_failure, &failure));
    let batches = writer.batches.borrow();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].origin, (0, 0));
    assert_eq!(batches[0].size, (1024, 256));
    assert_eq!(batches[0].bytes, band);
}

#[test]
fn early_eof_after_a_submitted_band_reports_the_consumed_prefix() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let band = vec![37; TEXTURE_UPLOAD_BATCH_BYTE_LIMIT];
    let source = stream::iter([Ok::<_, SourceFailure>(band.as_slice()), Ok(&[1, 2, 3][..])]);
    let mut writer = RecordingTextureBatchWriter::default();
    let result = block_on(feed_texture_stream(
        &operation,
        TextureStorageFormat::Rgba8Srgb,
        rgba_layout((1024, 257)),
        source,
        &mut writer,
    ));
    let Err(TextureStreamError::Backend(WgpuBackendError::Texture(
        TextureManagerError::InvalidPixels(PixelBufferError::StorageTooSmall {
            required,
            provided,
        }),
    ))) = result
    else {
        panic!("expected a short streamed prefix");
    };
    assert_eq!(required, TEXTURE_UPLOAD_BATCH_BYTE_LIMIT + 4096);
    assert_eq!(provided, TEXTURE_UPLOAD_BATCH_BYTE_LIMIT + 3);
    assert_eq!(writer.batches.borrow().len(), 1);
}

#[test]
fn complete_prefix_ignores_trailing_bytes_and_does_not_wait_for_eof() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let bytes = [12, 34, 56, 255, 99, 88];
    let source = stream::iter([Ok::<_, SourceFailure>(&bytes[..])]).chain(stream::pending());
    let mut writer = RecordingTextureBatchWriter::default();
    let (waker, _) = wake_channel();
    let mut context = Context::from_waker(&waker);
    {
        let mut upload = pin!(feed_texture_stream(
            &operation,
            TextureStorageFormat::Rgba8Srgb,
            rgba_layout((1, 1)),
            source,
            &mut writer,
        ));
        assert!(matches!(
            upload.as_mut().poll(&mut context),
            Poll::Ready(Ok(()))
        ));
    }
    let batches = writer.batches.borrow();
    assert_eq!(batches.len(), 1);
    assert_eq!(&batches[0].bytes[..4], &bytes[..4]);
    assert!(batches[0].bytes[4..].iter().all(|&byte| byte == 0));
}

#[test]
fn cancellation_wakes_a_driver_waiting_for_source_bytes() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let mut writer = RecordingTextureBatchWriter::default();
    let (waker, wakes) = wake_channel();
    let mut context = Context::from_waker(&waker);
    {
        let mut upload = pin!(feed_texture_stream(
            &operation,
            TextureStorageFormat::Rgba8Srgb,
            rgba_layout((1, 1)),
            stream::pending::<Result<&[u8], SourceFailure>>(),
            &mut writer,
        ));
        assert!(upload.as_mut().poll(&mut context).is_pending());
        operations.remove(42, || {});
        wakes.recv().unwrap();
        assert!(matches!(
            upload.as_mut().poll(&mut context),
            Poll::Ready(Err(TextureStreamError::Backend(WgpuBackendError::Texture(
                TextureManagerError::UploadCancelled
            ))))
        ));
    }
    assert!(writer.batches.borrow().is_empty());
}

#[test]
fn cancellation_wakes_a_driver_waiting_to_write_a_batch() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let (_resume_sender, resume_receiver) = oneshot::channel();
    let mut writer = RecordingTextureBatchWriter {
        resume: Some(resume_receiver),
        ..RecordingTextureBatchWriter::default()
    };
    let (waker, wakes) = wake_channel();
    let mut context = Context::from_waker(&waker);
    {
        let mut upload = pin!(feed_texture_stream(
            &operation,
            TextureStorageFormat::Rgba8Srgb,
            rgba_layout((1, 1)),
            stream::iter([Ok::<_, SourceFailure>(&[12, 34, 56, 255][..])]),
            &mut writer,
        ));
        assert!(upload.as_mut().poll(&mut context).is_pending());
        operations.remove(42, || {});
        wakes.recv().unwrap();
        assert!(matches!(
            upload.as_mut().poll(&mut context),
            Poll::Ready(Err(TextureStreamError::Backend(WgpuBackendError::Texture(
                TextureManagerError::UploadCancelled
            ))))
        ));
    }
    assert!(writer.batches.borrow().is_empty());
}

#[test]
fn device_loss_wakes_a_driver_waiting_for_source_bytes() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let mut writer = RecordingTextureBatchWriter::default();
    let (waker, wakes) = wake_channel();
    let mut context = Context::from_waker(&waker);
    {
        let mut upload = pin!(feed_texture_stream(
            &operation,
            TextureStorageFormat::Rgba8Srgb,
            rgba_layout((1, 1)),
            stream::pending::<Result<&[u8], SourceFailure>>(),
            &mut writer,
        ));
        assert!(upload.as_mut().poll(&mut context).is_pending());
        operations
            .failure_signal()
            .set(TextureTransferError::DeviceLost)
            .unwrap();
        wakes.recv().unwrap();
        assert!(matches!(
            upload.as_mut().poll(&mut context),
            Poll::Ready(Err(TextureStreamError::Backend(
                WgpuBackendError::DeviceLost
            )))
        ));
    }
    assert!(writer.batches.borrow().is_empty());
}

#[test]
fn writer_completion_wakes_the_driver_and_preserves_band_origins() {
    let operations = TextureUploadOperations::default();
    let operation = operations.begin(42).unwrap();
    let bytes = vec![37; TEXTURE_UPLOAD_BATCH_BYTE_LIMIT + 4096];
    let (resume_sender, resume_receiver) = oneshot::channel();
    let mut writer = RecordingTextureBatchWriter {
        resume: Some(resume_receiver),
        ..RecordingTextureBatchWriter::default()
    };
    let (waker, wakes) = wake_channel();
    let mut context = Context::from_waker(&waker);
    {
        let mut upload = pin!(feed_texture_stream(
            &operation,
            TextureStorageFormat::Rgba8Srgb,
            rgba_layout((1024, 257)),
            stream::iter([Ok::<_, SourceFailure>(bytes.as_slice())]),
            &mut writer,
        ));
        assert!(upload.as_mut().poll(&mut context).is_pending());
        resume_sender.send(()).unwrap();
        wakes.recv().unwrap();
        assert!(matches!(
            upload.as_mut().poll(&mut context),
            Poll::Ready(Ok(()))
        ));
    }
    let batches = writer.batches.borrow();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].origin, (0, 0));
    assert_eq!(batches[0].size, (1024, 256));
    assert_eq!(batches[1].origin, (0, 256));
    assert_eq!(batches[1].size, (1024, 1));
    assert_eq!(batches[1].bytes, bytes[TEXTURE_UPLOAD_BATCH_BYTE_LIMIT..]);
}
